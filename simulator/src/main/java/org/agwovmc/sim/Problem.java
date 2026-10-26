package org.agwovmc.sim;

import org.agwovmc.revision.Fitness;
import org.agwovmc.revision.Repair;

/**
 * Eq. (8) for one consolidation cycle. The demand snapshot and all normalisation constants are fixed while an
 * optimiser runs and do not depend on the incumbent placement (Assumption A1). Terms:
 *  E    = (P(x) - P_LB) / (P_full - P_LB), where
 *         P(x)   = power of the active hosts (Eq. 1; above u = 1 the curve is continued with its slope at u = 1),
 *         P_LB   = relaxation lower bound: idle power of the cheapest fractional cover of the CPU demand (idle W per MIPS,
 *                  at least ceil(RAM / 8 GB) hosts) plus total demand x the smallest dynamic slope (peak - idle) / MIPS;
 *                  valid because 2u - u^1.4 >= u on [0, 1],
 *         P_full = power with every host on and the demand spread in proportion to capacity (no consolidation).
 *  SLA  = overload risk: sum over active hosts of max(0, u - 0.8), divided by 0.2 x k_LB, where k_LB is the smallest
 *         number of hosts that can carry the total CPU and RAM demand.
 *  Emig = RAM of VMs moved / total RAM (fraction of the worst case in which every VM moves; Eq. 6 is proportional to it).
 *  U    = coefficient of variation of active-host utilisation (dimensionless, not rescaled).
 * Capacity violations (CPU demand > MIPS capacity, RAM > 8 GB) add a penalty of 10 x the relative excess.
 */
public final class Problem implements Fitness, Repair, org.agwovmc.revision.LocalMoves {
    public static final double RAM_CAP = 8192;
    final int n, m;
    final double[] cap, dem, ram;
    final int[] hostType, cur;
    final double[] w;
    final double totalRam, totalDem;
    final double eLb, eFull, riskDen;
    final int kLb;

    public Problem(double[] cap, int[] hostType, double[] dem, double[] ram, int[] cur, double[] weights) {
        this.cap = cap; this.hostType = hostType; this.dem = dem; this.ram = ram; this.cur = cur; this.w = weights;
        n = dem.length; m = cap.length;
        double tr = 0, td = 0; for (double x : ram) tr += x; for (double d : dem) td += d; totalRam = tr; totalDem = td;
        eLb = lowerBound();
        eFull = fullPower();
        kLb = minHosts();
        riskDen = 0.2 * Math.max(1, kLb);
    }

    /** Power used inside the fitness: Eq. (1) on [0, 1], continued linearly with slope 0.6 (P_max - P_idle) above u = 1. */
    static double fitnessWatts(int type, double u) {
        if (u <= 1.0) return Power.watts(type, u);
        return Power.MAX[type] + 0.6 * (Power.MAX[type] - Power.IDLE[type]) * (u - 1.0);
    }

    /** Raw terms: power (W), overload risk, moved-RAM fraction, CV of utilisation, violation, active hosts. */
    double[] raw(int[] p) {
        double[] load = new double[m], ru = new double[m]; int[] cnt = new int[m];
        double movedRam = 0;
        for (int j = 0; j < n; j++) {
            int h = p[j]; load[h] += dem[j]; ru[h] += ram[j]; cnt[h]++;
            if (cur[j] >= 0 && cur[j] != h) movedRam += ram[j];
        }
        double e = 0, risk = 0, viol = 0, s = 0, s2 = 0; int act = 0;
        for (int h = 0; h < m; h++) {
            if (cnt[h] == 0) continue;
            double u = load[h] / cap[h];
            act++; s += u; s2 += u * u;
            e += fitnessWatts(hostType[h], u);
            risk += Math.max(0, u - 0.8);
            viol += Math.max(0, u - 1.0) + Math.max(0, ru[h] / RAM_CAP - 1.0);
        }
        double mean = s / Math.max(1, act), var = Math.max(0, s2 / Math.max(1, act) - mean * mean);
        double cv = mean > 0 ? Math.sqrt(var) / mean : 0;
        return new double[]{e, risk, movedRam / totalRam, cv, viol, act};
    }

    double energyTerm(double watts) { return (watts - eLb) / (eFull - eLb); }

    @Override public double[] objectives(int[] p) {
        double[] o = raw(p);
        return new double[]{energyTerm(o[0]), o[1] / riskDen, o[2], o[3]};
    }

    @Override public double[] objectivesAndViolation(int[] p) {
        double[] o = raw(p);
        return new double[]{energyTerm(o[0]), o[1] / riskDen, o[2], o[3], o[4]};
    }

    /** The same demand snapshot and incumbent with another weight vector (used to trace the weighted-sum points). */
    public Problem withWeights(double[] weights) { return new Problem(cap, hostType, dem, ram, cur, weights); }

    @Override public double evaluate(int[] p) {
        double[] o = raw(p);
        return w[0] * energyTerm(o[0]) + w[1] * o[1] / riskDen + w[2] * o[2] + w[3] * o[3] + 10.0 * o[4];
    }

    /** Relaxation lower bound on the power of any placement that carries the current demand (see class comment). */
    double lowerBound() {
        Integer[] o = new Integer[m]; for (int h = 0; h < m; h++) o[h] = h;
        java.util.Arrays.sort(o, (a, b) -> Double.compare(Power.IDLE[hostType[a]] / cap[a], Power.IDLE[hostType[b]] / cap[b]));
        double idle = 0, need = totalDem;
        for (int i = 0; i < m && need > 1e-12; i++) {
            int h = o[i]; double take = Math.min(1.0, need / cap[h]);
            idle += take * Power.IDLE[hostType[h]]; need -= take * cap[h];
        }
        double minIdle = Double.MAX_VALUE, slope = Double.MAX_VALUE;
        for (int h = 0; h < m; h++) {
            minIdle = Math.min(minIdle, Power.IDLE[hostType[h]]);
            slope = Math.min(slope, (Power.MAX[hostType[h]] - Power.IDLE[hostType[h]]) / cap[h]);
        }
        double idleRam = Math.ceil(totalRam / RAM_CAP - 1e-9) * minIdle;
        return Math.max(idle, idleRam) + totalDem * slope;
    }

    /** Reference power without consolidation: every host on, demand spread in proportion to capacity. */
    double fullPower() {
        double C = 0; for (double c : cap) C += c;
        double u = totalDem / C, e = 0;
        for (int h = 0; h < m; h++) e += fitnessWatts(hostType[h], u);
        return e;
    }

    /** Smallest number of hosts (largest first) whose CPU and RAM capacity covers the total demand. */
    int minHosts() {
        double[] c = cap.clone(); java.util.Arrays.sort(c);
        double sc = 0, sr = 0; int k = 0;
        for (int i = c.length - 1; i >= 0 && (sc < totalDem || sr < totalRam); i--) { sc += c[i]; sr += RAM_CAP; k++; }
        return k;
    }

    /** Host evacuation: move every VM of one of the five least-loaded active hosts to other active hosts (best fit, <= 90% CPU). */
    @Override public int[] propose(int[] p, java.util.Random rng) {
        double[] load = new double[m], ru = new double[m]; int[] cnt = new int[m];
        for (int j = 0; j < n; j++) { load[p[j]] += dem[j]; ru[p[j]] += ram[j]; cnt[p[j]]++; }
        int[] low = new int[5]; double[] lu = new double[5]; int nl = 0;
        for (int h = 0; h < m; h++) {
            if (cnt[h] == 0) continue;
            double u = load[h] / cap[h];
            if (nl < 5) { low[nl] = h; lu[nl] = u; nl++; }
            else { int w = 0; for (int i = 1; i < 5; i++) if (lu[i] > lu[w]) w = i; if (u < lu[w]) { low[w] = h; lu[w] = u; } }
        }
        if (nl < 2) return null;
        int src = low[rng.nextInt(nl)];
        int[] q = p.clone();
        for (int j = 0; j < n; j++) {
            if (q[j] != src) continue;
            int best = -1; double bu = -1;
            for (int h = 0; h < m; h++) {
                if (h == src || cnt[h] == 0) continue;
                if (load[h] + dem[j] > 0.9 * cap[h] || ru[h] + ram[j] > RAM_CAP) continue;
                double u = load[h] / cap[h];
                if (u > bu) { bu = u; best = h; }
            }
            if (best < 0) return null;
            q[j] = best; load[best] += dem[j]; ru[best] += ram[j];
        }
        return q;
    }

    @Override public int numVms() { return n; }
    @Override public int numPms() { return m; }

    @Override public boolean isFeasible(int[] p) {
        double[] load = new double[m], ru = new double[m];
        for (int j = 0; j < n; j++) { load[p[j]] += dem[j]; ru[p[j]] += ram[j]; }
        for (int h = 0; h < m; h++) if (load[h] > cap[h] + 1e-9 || ru[h] > RAM_CAP + 1e-9) return false;
        return true;
    }

    /** Greedy repair: move the largest VMs off over-capacity hosts to the least-loaded host that can take them. */
    @Override public boolean repair(int[] p) {
        double[] load = new double[m], ru = new double[m];
        for (int j = 0; j < n; j++) { load[p[j]] += dem[j]; ru[p[j]] += ram[j]; }
        boolean changed = false;
        for (int j = 0; j < n; j++) {
            int h = p[j];
            if (load[h] <= cap[h] + 1e-9 && ru[h] <= RAM_CAP + 1e-9) continue;
            int best = -1; double bu = Double.MAX_VALUE; int seen = 0;
            int start = (int) ((j * 2654435761L) % m);
            for (int k = 0; k < m && seen < 20; k++) {
                int q = (start + k) % m;
                if (q == h) continue;
                if (load[q] + dem[j] > cap[q] || ru[q] + ram[j] > RAM_CAP) continue;
                seen++;
                double u = load[q] / cap[q];
                if (u < bu) { bu = u; best = q; }
            }
            if (best < 0) continue;
            load[h] -= dem[j]; ru[h] -= ram[j]; load[best] += dem[j]; ru[best] += ram[j];
            p[j] = best; changed = true;
        }
        return changed;
    }
}
