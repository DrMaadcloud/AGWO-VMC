package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Real-coded GA (tournament selection, blend crossover 0.8, Gaussian mutation 0.05, elitism) on the shared encoding. */
public final class GaOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private final double pc = 0.8, pm = 0.05;
    private LocalMoves lsMoves = null; private int tLs = 5, lMax = 50;
    /** Adds the AGWO-VMC local search (swaps + host evacuation) every tLs iterations, for the like-for-like "+LS" baseline. */
    public GaOptimizer withLocalSearch(LocalMoves evacuation, int tLs, int lMax) { this.lsMoves = evacuation; this.tLs = tLs; this.lMax = lMax; return this; }
    public GaOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) {
        N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta;
    }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        int n = f.numVms(), m = f.numPms();
        EvalBudget budget = new EvalBudget(maxEvals);
        CandidateSoftmaxDecoder dec = new CandidateSoftmaxDecoder(m, c, beta);
        ElitistArchive arch = new ElitistArchive();
        double[][] X = new double[N][n]; double[] F = new double[N];
        int[][] P = new int[N][];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < n; j++) X[i][j] = Init.coord(incumbent, i, j, m, rng);
            P[i] = dec.decode(X[i], rng); repair.repair(P[i]);
            if (!budget.take()) break;
            F[i] = f.evaluate(P[i]); arch.offer(P[i], F[i]);
        }
        if (incumbent != null && budget.take()) { int[] inc = incumbent.clone(); repair.repair(inc); arch.offer(inc, f.evaluate(inc)); }
        List<Double> hist = new ArrayList<>(); hist.add(arch.fitness());
        int t = 0;
        while (!budget.exhausted()) {
            t++;
            int best = 0; for (int i = 1; i < N; i++) if (F[i] < F[best]) best = i;
            double[][] Y = new double[N][]; double[] FY = new double[N]; int[][] PY = new int[N][];
            Y[0] = X[best].clone(); FY[0] = F[best]; PY[0] = P[best];
            int repaired = 0, evaluated = 0;
            for (int i = 1; i < N && !budget.exhausted(); i++) {
                double[] a = X[tour(F, rng)], b = X[tour(F, rng)];
                double[] child = new double[n];
                boolean cross = rng.nextDouble() < pc;
                for (int j = 0; j < n; j++) {
                    double v = cross ? (rng.nextBoolean() ? a[j] : b[j]) : a[j];
                    if (rng.nextDouble() < pm) v += rng.nextGaussian() * 0.1 * m;
                    child[j] = Math.min(m - 1, Math.max(0, v));
                }
                int[] p = dec.decode(child, rng);
                if (repair.repair(p)) repaired++;
                if (!budget.take()) break;
                Y[i] = child; PY[i] = p; FY[i] = f.evaluate(p); evaluated++;
                arch.offer(p, FY[i]);
            }
            for (int i = 0; i < N; i++) if (Y[i] != null) { X[i] = Y[i]; F[i] = FY[i]; P[i] = PY[i]; }
            if (lsMoves != null && t % tLs == 0) LocalSearch.apply(arch, f, repair, lMax, lsMoves, budget, rng);
            hist.add(arch.fitness());
            if (log != null) log.onIteration(t, arch.fitness(), Double.NaN, Double.NaN, Double.NaN,
                    Diversity.continuous(X), Diversity.placement(P), evaluated == 0 ? 0 : (double) repaired / evaluated);
        }
        return new OptimizerResult(arch.best(), arch.fitness(), t, budget.used(), hist);
    }
    private int tour(double[] F, Random rng) { int a = rng.nextInt(N), b = rng.nextInt(N); return F[a] <= F[b] ? a : b; }
}
