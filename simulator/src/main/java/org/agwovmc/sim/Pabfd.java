package org.agwovmc.sim;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * BFD baseline: power-aware best-fit decreasing with overload/underload migration (Beloglazov and Buyya [3]).
 * VMs are moved off hosts above 90% CPU (smallest RAM first) and the least-loaded hosts are evacuated when their
 * VMs fit elsewhere; each VM goes to the host with the smallest increase in power (Eq. 1) that stays <= 90%.
 */
public final class Pabfd {
    public static int[] place(Problem pr) {
        int n = pr.n, m = pr.m;
        int[] p = pr.cur.clone();
        double[] load = new double[m], ru = new double[m];
        List<List<Integer>> on = new ArrayList<>();
        for (int h = 0; h < m; h++) on.add(new ArrayList<>());
        for (int j = 0; j < n; j++) { load[p[j]] += pr.dem[j]; ru[p[j]] += pr.ram[j]; on.get(p[j]).add(j); }
        List<Integer> toPlace = new ArrayList<>();
        boolean[] excluded = new boolean[m];
        for (int h = 0; h < m; h++) {
            if (load[h] <= 0.9 * pr.cap[h]) continue;
            List<Integer> vms = new ArrayList<>(on.get(h));
            vms.sort(Comparator.comparingDouble(j -> pr.ram[j]));
            for (int j : vms) {
                if (load[h] <= 0.9 * pr.cap[h]) break;
                load[h] -= pr.dem[j]; ru[h] -= pr.ram[j]; on.get(h).remove((Integer) j); toPlace.add(j);
            }
            excluded[h] = true;
        }
        toPlace.sort((a, b) -> Double.compare(pr.dem[b], pr.dem[a]));
        for (int j : toPlace) {
            int h = best(pr, load, ru, j, excluded, 0.9);
            if (h < 0) h = best(pr, load, ru, j, new boolean[m], 1.0);
            if (h < 0) h = pr.cur[j];
            p[j] = h; load[h] += pr.dem[j]; ru[h] += pr.ram[j]; on.get(h).add(j);
        }
        // underload: try to evacuate up to three of the least-loaded active hosts
        for (int k = 0; k < 3; k++) {
            int src = -1; double su = Double.MAX_VALUE;
            for (int h = 0; h < m; h++)
                if (!on.get(h).isEmpty() && !excluded[h] && load[h] / pr.cap[h] < su) { su = load[h] / pr.cap[h]; src = h; }
            if (src < 0) break;
            excluded[src] = true;
            double[] l2 = load.clone(), r2 = ru.clone();
            int[] target = new int[on.get(src).size()]; boolean ok = true; int i = 0;
            List<Integer> vms = new ArrayList<>(on.get(src));
            vms.sort((a, b) -> Double.compare(pr.dem[b], pr.dem[a]));
            boolean[] ex2 = excluded.clone();
            for (int h = 0; h < m; h++) if (on.get(h).isEmpty()) ex2[h] = true;   // do not wake idle hosts
            for (int j : vms) {
                int h = best(pr, l2, r2, j, ex2, 0.9);
                if (h < 0) { ok = false; break; }
                target[i++] = h; l2[h] += pr.dem[j]; r2[h] += pr.ram[j];
            }
            if (!ok) continue;
            i = 0;
            for (int j : vms) { int h = target[i++]; p[j] = h; on.get(h).add(j); }
            on.get(src).clear(); load = l2; ru = r2; load[src] = 0; ru[src] = 0;
        }
        return p;
    }

    private static int best(Problem pr, double[] load, double[] ru, int j, boolean[] excluded, double thr) {
        int best = -1; double bd = Double.MAX_VALUE;
        for (int h = 0; h < pr.m; h++) {
            if (excluded[h]) continue;
            if (load[h] + pr.dem[j] > thr * pr.cap[h] || ru[h] + pr.ram[j] > Problem.RAM_CAP) continue;
            double before = load[h] > 0 ? Power.watts(pr.hostType[h], load[h] / pr.cap[h]) : 0;
            double d = Power.watts(pr.hostType[h], (load[h] + pr.dem[j]) / pr.cap[h]) - before;
            if (d < bd) { bd = d; best = h; }
        }
        return best;
    }
}
