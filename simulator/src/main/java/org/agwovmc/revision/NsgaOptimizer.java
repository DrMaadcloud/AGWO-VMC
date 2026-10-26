package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * NSGA-II (Deb et al. [44]) on the shared encoding, repair, warm start and evaluation budget. The four objectives are the
 * normalised terms of Eq. (8) (energy, SLA risk, migration, imbalance); capacity violations are handled by constrained
 * domination (a feasible solution dominates an infeasible one; infeasible ones are compared on the violation).
 * Variation: binary tournament on (rank, crowding), simulated binary crossover (p_c = 0.9, eta_c = 20) and polynomial
 * mutation (p_m = 1/n, eta_m = 20) on the continuous PM coordinates, followed by the softmax decoding and repair used by
 * every method. One evaluation of the objectives is charged per decoded placement.
 * With withLocalSearch(...) the AGWO-VMC moves (pairwise swaps and host evacuation) are applied every T_ls generations to
 * randomly chosen members of the first front, and the resulting placements join the offspring (memetic NSGA-II, "+LS").
 * A single placement is returned for execution: the knee point of the final first front (closest to the utopia point after
 * utopia/nadir normalisation of the front) or the AHP-best member (lowest weighted sum, Eq. 8).
 */
public final class NsgaOptimizer {
    public enum Pick { KNEE, AHP }

    private final int N; private final long maxEvals; private final int c; private final double beta; private final Pick pick;
    private final double pc = 0.9, etaC = 20, etaM = 20;
    private LocalMoves lsMoves = null; private boolean useLs = false; private int tLs = 5, lMax = 50;
    private List<double[]> front = new ArrayList<>();

    public NsgaOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta, Pick pick) {
        N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta; this.pick = pick;
    }
    public NsgaOptimizer withLocalSearch(LocalMoves evacuation, int tLs, int lMax) {
        this.useLs = evacuation != null; this.lsMoves = evacuation; this.tLs = tLs; this.lMax = lMax; return this;
    }
    /** Objective vectors (4 normalised terms) of the distinct feasible members of the final first front. */
    public List<double[]> frontObjectives() { return front; }

    private static final class Ind {
        double[] x; int[] p; double[] o; double v, fw; int rank; double crowd;
    }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        final int n = f.numVms(), m = f.numPms();
        final double pm = 1.0 / n;
        EvalBudget budget = new EvalBudget(maxEvals);
        CandidateSoftmaxDecoder dec = new CandidateSoftmaxDecoder(m, c, beta);
        List<Ind> P = new ArrayList<>();
        if (incumbent != null && budget.take()) {
            int[] inc = incumbent.clone(); repair.repair(inc);
            P.add(make(HhoOptimizer.toD(inc), inc, f));
        }
        for (int i = P.size(); i < N && !budget.exhausted(); i++) {
            double[] x = new double[n];
            for (int j = 0; j < n; j++) x[j] = Init.coord(incumbent, i, j, m, rng);
            int[] p = dec.decode(x, rng); repair.repair(p);
            if (!budget.take()) break;
            P.add(make(x, p, f));
        }
        rankAndCrowd(P);
        List<Double> hist = new ArrayList<>(); hist.add(minFw(P));
        int t = 0;
        while (!budget.exhausted()) {
            t++;
            List<Ind> Q = new ArrayList<>();
            int repaired = 0, evaluated = 0;
            while (Q.size() < N && !budget.exhausted()) {
                Ind a = tour(P, rng), b = tour(P, rng);
                double[][] ch = sbx(a.x, b.x, m - 1, rng);
                for (double[] y : ch) {
                    if (Q.size() >= N || budget.exhausted()) break;
                    mutate(y, m - 1, pm, rng);
                    int[] p = dec.decode(y, rng);
                    if (repair.repair(p)) repaired++;
                    if (!budget.take()) break;
                    Q.add(make(y, p, f)); evaluated++;
                }
            }
            if (useLs && t % tLs == 0) {
                List<Ind> F1 = new ArrayList<>();
                for (Ind q : P) if (q.rank == 0) F1.add(q);
                for (int e = 0; e < lMax && !budget.exhausted(); e++) {
                    int[] base = F1.get(rng.nextInt(F1.size())).p;
                    int[] q;
                    if (e % 2 == 0) {                       // pairwise swap of two VMs on different PMs
                        q = base.clone();
                        int i = rng.nextInt(n), j = rng.nextInt(n);
                        if (i == j || q[i] == q[j]) continue;
                        int h = q[i]; q[i] = q[j]; q[j] = h;
                        if (!repair.isFeasible(q)) continue;
                    } else {                                // host evacuation
                        q = lsMoves.propose(base, rng);
                        if (q == null) continue;
                    }
                    if (!budget.take()) break;
                    Q.add(make(HhoOptimizer.toD(q), q, f));
                }
            }
            List<Ind> R = new ArrayList<>(P); R.addAll(Q);
            P = select(R, N);
            hist.add(minFw(P));
            if (log != null) {
                double[][] X = new double[P.size()][]; int[][] D = new int[P.size()][];
                for (int i = 0; i < P.size(); i++) { X[i] = P.get(i).x; D[i] = P.get(i).p; }
                log.onIteration(t, minFw(P), Double.NaN, Double.NaN, Double.NaN, Diversity.continuous(X), Diversity.placement(D),
                        evaluated == 0 ? 0 : (double) repaired / evaluated);
            }
        }
        // final first front (feasible members only when any exist)
        List<Ind> F1 = new ArrayList<>();
        for (Ind q : P) if (q.rank == 0) F1.add(q);
        List<Ind> feas = new ArrayList<>();
        for (Ind q : F1) if (q.v <= 1e-12) feas.add(q);
        if (!feas.isEmpty()) F1 = feas;
        front = new ArrayList<>();
        List<int[]> seen = new ArrayList<>();
        for (Ind q : F1) {
            boolean dup = false;
            for (int[] s : seen) if (Arrays.equals(s, q.p)) { dup = true; break; }
            if (!dup) { seen.add(q.p); front.add(q.o.clone()); }
        }
        Ind chosen = F1.get(0);
        if (pick == Pick.AHP) {
            for (Ind q : F1) if (q.fw < chosen.fw) chosen = q;
        } else {
            double[] lo = new double[4], hi = new double[4];
            Arrays.fill(lo, Double.POSITIVE_INFINITY); Arrays.fill(hi, Double.NEGATIVE_INFINITY);
            for (Ind q : F1) for (int k = 0; k < 4; k++) { lo[k] = Math.min(lo[k], q.o[k]); hi[k] = Math.max(hi[k], q.o[k]); }
            double bd = Double.POSITIVE_INFINITY;
            for (Ind q : F1) {
                double d = 0;
                for (int k = 0; k < 4; k++) { double r = hi[k] - lo[k]; if (r > 1e-12) { double z = (q.o[k] - lo[k]) / r; d += z * z; } }
                if (d < bd - 1e-15 || (Math.abs(d - bd) <= 1e-15 && q.fw < chosen.fw)) { bd = d; chosen = q; }
            }
        }
        return new OptimizerResult(chosen.p.clone(), chosen.fw, t, budget.used(), hist);
    }

    private static Ind make(double[] x, int[] p, Fitness f) {
        Ind q = new Ind(); q.x = x; q.p = p;
        double[] ov = f.objectivesAndViolation(p);
        q.o = Arrays.copyOf(ov, 4); q.v = ov[4];
        q.fw = f.evaluate(p);   // weighted sum (Eq. 8), kept for the AHP-best choice; same placement, not a new search point
        return q;
    }
    private static double minFw(List<Ind> P) { double b = Double.POSITIVE_INFINITY; for (Ind q : P) b = Math.min(b, q.fw); return b; }

    /** Constrained domination (Deb): feasibility first, then Pareto dominance on the objectives. */
    private static boolean dominates(Ind a, Ind b) {
        boolean fa = a.v <= 1e-12, fb = b.v <= 1e-12;
        if (fa && !fb) return true;
        if (!fa && !fb) return a.v < b.v;
        if (!fa) return false;
        boolean strict = false;
        for (int k = 0; k < 4; k++) {
            if (a.o[k] > b.o[k] + 1e-15) return false;
            if (a.o[k] < b.o[k] - 1e-15) strict = true;
        }
        return strict;
    }

    private static List<List<Ind>> sort(List<Ind> R) {
        int s = R.size();
        int[] cnt = new int[s]; List<List<Integer>> dom = new ArrayList<>();
        for (int i = 0; i < s; i++) dom.add(new ArrayList<>());
        List<List<Ind>> fronts = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i < s; i++) {
            for (int j = i + 1; j < s; j++) {
                if (dominates(R.get(i), R.get(j))) { dom.get(i).add(j); cnt[j]++; }
                else if (dominates(R.get(j), R.get(i))) { dom.get(j).add(i); cnt[i]++; }
            }
        }
        for (int i = 0; i < s; i++) if (cnt[i] == 0) cur.add(i);
        int r = 0;
        while (!cur.isEmpty()) {
            List<Ind> fr = new ArrayList<>(); List<Integer> next = new ArrayList<>();
            for (int i : cur) {
                R.get(i).rank = r; fr.add(R.get(i));
                for (int j : dom.get(i)) if (--cnt[j] == 0) next.add(j);
            }
            fronts.add(fr); cur = next; r++;
        }
        return fronts;
    }

    private static void crowding(List<Ind> F) {
        for (Ind q : F) q.crowd = 0;
        if (F.size() <= 2) { for (Ind q : F) q.crowd = Double.POSITIVE_INFINITY; return; }
        for (int k = 0; k < 4; k++) {
            final int kk = k;
            F.sort(Comparator.comparingDouble(q -> q.o[kk]));
            double r = F.get(F.size() - 1).o[k] - F.get(0).o[k];
            F.get(0).crowd = Double.POSITIVE_INFINITY; F.get(F.size() - 1).crowd = Double.POSITIVE_INFINITY;
            if (r <= 1e-15) continue;
            for (int i = 1; i < F.size() - 1; i++) F.get(i).crowd += (F.get(i + 1).o[k] - F.get(i - 1).o[k]) / r;
        }
    }

    private static void rankAndCrowd(List<Ind> P) { for (List<Ind> F : sort(P)) crowding(F); }

    private static List<Ind> select(List<Ind> R, int N) {
        List<Ind> out = new ArrayList<>();
        for (List<Ind> F : sort(R)) {
            crowding(F);
            if (out.size() + F.size() <= N) { out.addAll(F); continue; }
            F.sort((a, b) -> Double.compare(b.crowd, a.crowd));
            for (Ind q : F) { if (out.size() >= N) break; out.add(q); }
            break;
        }
        return out;
    }

    private static Ind tour(List<Ind> P, Random rng) {
        Ind a = P.get(rng.nextInt(P.size())), b = P.get(rng.nextInt(P.size()));
        if (a.rank != b.rank) return a.rank < b.rank ? a : b;
        if (a.crowd != b.crowd) return a.crowd > b.crowd ? a : b;
        return rng.nextBoolean() ? a : b;
    }

    /** Bounded simulated binary crossover on [0, ub]. */
    private double[][] sbx(double[] p1, double[] p2, double ub, Random rng) {
        int n = p1.length;
        double[] c1 = p1.clone(), c2 = p2.clone();
        if (rng.nextDouble() > pc) return new double[][]{c1, c2};
        for (int j = 0; j < n; j++) {
            if (rng.nextDouble() > 0.5 || Math.abs(p1[j] - p2[j]) < 1e-14) continue;
            double y1 = Math.min(p1[j], p2[j]), y2 = Math.max(p1[j], p2[j]);
            double u = rng.nextDouble();
            double b1 = 1 + 2 * (y1 - 0) / (y2 - y1), a1 = 2 - Math.pow(b1, -(etaC + 1));
            double bq1 = u <= 1 / a1 ? Math.pow(u * a1, 1 / (etaC + 1)) : Math.pow(1 / (2 - u * a1), 1 / (etaC + 1));
            double b2 = 1 + 2 * (ub - y2) / (y2 - y1), a2 = 2 - Math.pow(b2, -(etaC + 1));
            double bq2 = u <= 1 / a2 ? Math.pow(u * a2, 1 / (etaC + 1)) : Math.pow(1 / (2 - u * a2), 1 / (etaC + 1));
            double v1 = 0.5 * ((y1 + y2) - bq1 * (y2 - y1)), v2 = 0.5 * ((y1 + y2) + bq2 * (y2 - y1));
            v1 = Math.min(ub, Math.max(0, v1)); v2 = Math.min(ub, Math.max(0, v2));
            if (rng.nextBoolean()) { c1[j] = v2; c2[j] = v1; } else { c1[j] = v1; c2[j] = v2; }
        }
        return new double[][]{c1, c2};
    }

    /** Bounded polynomial mutation on [0, ub]. */
    private void mutate(double[] y, double ub, double pm, Random rng) {
        for (int j = 0; j < y.length; j++) {
            if (rng.nextDouble() >= pm) continue;
            double x = y[j], d1 = x / ub, d2 = (ub - x) / ub, u = rng.nextDouble(), dq;
            if (u < 0.5) dq = Math.pow(2 * u + (1 - 2 * u) * Math.pow(1 - d1, etaM + 1), 1 / (etaM + 1)) - 1;
            else dq = 1 - Math.pow(2 * (1 - u) + 2 * (u - 0.5) * Math.pow(1 - d2, etaM + 1), 1 / (etaM + 1));
            y[j] = Math.min(ub, Math.max(0, x + dq * ub));
        }
    }
}
