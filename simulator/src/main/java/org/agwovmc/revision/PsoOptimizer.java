package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** PSO with inertia 0.9 -> 0.4 and c1 = c2 = 2.0 on the shared encoding. */
public final class PsoOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private LocalMoves lsMoves = null; private int tLs = 5, lMax = 50;
    /** Adds the AGWO-VMC local search (swaps + host evacuation) every tLs iterations, for the like-for-like "+LS" baseline. */
    public PsoOptimizer withLocalSearch(LocalMoves evacuation, int tLs, int lMax) { this.lsMoves = evacuation; this.tLs = tLs; this.lMax = lMax; return this; }
    public PsoOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) {
        N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta;
    }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        int n = f.numVms(), m = f.numPms();
        EvalBudget budget = new EvalBudget(maxEvals);
        CandidateSoftmaxDecoder dec = new CandidateSoftmaxDecoder(m, c, beta);
        ElitistArchive arch = new ElitistArchive();
        double[][] X = new double[N][n], V = new double[N][n], Pb = new double[N][];
        double[] Fb = new double[N]; int[][] P = new int[N][];
        double vmax = 0.2 * m;
        int T = (int) Math.max(1, (maxEvals - N) / N);
        double[] g = null; double fg = Double.POSITIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < n; j++) { X[i][j] = Init.coord(incumbent, i, j, m, rng); V[i][j] = (rng.nextDouble() - 0.5) * vmax; }
            P[i] = dec.decode(X[i], rng); repair.repair(P[i]);
            if (!budget.take()) break;
            double fi = f.evaluate(P[i]); arch.offer(P[i], fi);
            Pb[i] = X[i].clone(); Fb[i] = fi;
            if (fi < fg) { fg = fi; g = X[i].clone(); }
        }
        if (incumbent != null && budget.take()) { int[] inc = incumbent.clone(); repair.repair(inc); arch.offer(inc, f.evaluate(inc)); }
        List<Double> hist = new ArrayList<>(); hist.add(arch.fitness());
        int t = 0;
        while (!budget.exhausted()) {
            t++;
            double w = 0.9 - 0.5 * Math.min(1.0, (double) t / T);
            int repaired = 0, evaluated = 0;
            for (int i = 0; i < N && !budget.exhausted(); i++) {
                for (int j = 0; j < n; j++) {
                    V[i][j] = w * V[i][j] + 2.0 * rng.nextDouble() * (Pb[i][j] - X[i][j]) + 2.0 * rng.nextDouble() * (g[j] - X[i][j]);
                    V[i][j] = Math.max(-vmax, Math.min(vmax, V[i][j]));
                    X[i][j] = Math.min(m - 1, Math.max(0, X[i][j] + V[i][j]));
                }
                P[i] = dec.decode(X[i], rng);
                if (repair.repair(P[i])) repaired++;
                if (!budget.take()) break;
                double fi = f.evaluate(P[i]); evaluated++;
                arch.offer(P[i], fi);
                if (fi < Fb[i]) { Fb[i] = fi; Pb[i] = X[i].clone(); }
                if (fi < fg) { fg = fi; g = X[i].clone(); }
            }
            if (lsMoves != null && t % tLs == 0) LocalSearch.apply(arch, f, repair, lMax, lsMoves, budget, rng);
            hist.add(arch.fitness());
            if (log != null) log.onIteration(t, arch.fitness(), Double.NaN, Double.NaN, Double.NaN,
                    Diversity.continuous(X), Diversity.placement(P), evaluated == 0 ? 0 : (double) repaired / evaluated);
        }
        return new OptimizerResult(arch.best(), arch.fitness(), t, budget.used(), hist);
    }
}
