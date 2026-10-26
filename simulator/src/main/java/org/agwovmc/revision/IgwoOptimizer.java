package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * I-GWO [21] (dimension learning-based hunting) on the shared encoding, repair and budget.
 * Each wolf proposes a GWO move and a DLH move and keeps the better of the two if it improves on itself.
 */
public final class IgwoOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private LocalMoves lsMoves = null; private int tLs = 5, lMax = 50;
    /** Adds the AGWO-VMC local search (swaps + host evacuation) every tLs iterations, for the like-for-like "+LS" baseline. */
    public IgwoOptimizer withLocalSearch(LocalMoves evacuation, int tLs, int lMax) { this.lsMoves = evacuation; this.tLs = tLs; this.lMax = lMax; return this; }
    public IgwoOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) {
        N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta;
    }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        int n = f.numVms(), m = f.numPms();
        EvalBudget budget = new EvalBudget(maxEvals);
        CandidateSoftmaxDecoder dec = new CandidateSoftmaxDecoder(m, c, beta);
        ElitistArchive arch = new ElitistArchive();
        double[][] X = new double[N][n]; double[] F = new double[N];
        int T = (int) Math.max(1, (maxEvals - N) / (2.0 * N));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < n; j++) X[i][j] = Init.coord(incumbent, i, j, m, rng);
            F[i] = HhoOptimizer.eval(X[i], f, repair, dec, budget, arch, rng);
        }
        if (incumbent != null && budget.take()) { int[] inc = incumbent.clone(); repair.repair(inc); arch.offer(inc, f.evaluate(inc)); }
        List<Double> hist = new ArrayList<>(); hist.add(arch.fitness());
        int t = 0;
        while (!budget.exhausted()) {
            t++;
            double a = 2.0 * (1.0 - Math.min(1.0, (double) t / T));
            int[] ord = order(F);
            double[][] L = {X[ord[0]], X[ord[1]], X[ord[2]]};
            double[][] Xn = new double[N][];
            double[] Fn = new double[N];
            for (int i = 0; i < N && !budget.exhausted(); i++) {
                double[] g = new double[n];
                for (int j = 0; j < n; j++) {
                    double s = 0;
                    for (int l = 0; l < 3; l++) {
                        double A = 2 * a * rng.nextDouble() - a, C = 2 * rng.nextDouble();
                        s += L[l][j] - A * Math.abs(C * L[l][j] - X[i][j]);
                    }
                    g[j] = s / 3;
                }
                HhoOptimizer.clamp(g, m);
                double R = dist(X[i], g);
                List<Integer> nb = new ArrayList<>();
                for (int q = 0; q < N; q++) if (dist(X[i], X[q]) <= R) nb.add(q);
                double[] h = new double[n];
                for (int j = 0; j < n; j++) {
                    double[] xn = X[nb.get(rng.nextInt(nb.size()))], xr = X[rng.nextInt(N)];
                    h[j] = X[i][j] + rng.nextDouble() * (xn[j] - xr[j]);
                }
                HhoOptimizer.clamp(h, m);
                double fg = HhoOptimizer.eval(g, f, repair, dec, budget, arch, rng);
                double fh = HhoOptimizer.eval(h, f, repair, dec, budget, arch, rng);
                double[] best = fg <= fh ? g : h; double fb = Math.min(fg, fh);
                if (fb < F[i]) { Xn[i] = best; Fn[i] = fb; } else { Xn[i] = X[i]; Fn[i] = F[i]; }
            }
            for (int i = 0; i < N; i++) if (Xn[i] != null) { X[i] = Xn[i]; F[i] = Fn[i]; }
            if (lsMoves != null && t % tLs == 0) LocalSearch.apply(arch, f, repair, lMax, lsMoves, budget, rng);
            hist.add(arch.fitness());
            if (log != null) log.onIteration(t, arch.fitness(), a, Double.NaN, Double.NaN, Diversity.continuous(X), Double.NaN, Double.NaN);
        }
        return new OptimizerResult(arch.best(), arch.fitness(), t, budget.used(), hist);
    }
    private static double dist(double[] a, double[] b) { double s = 0; for (int j = 0; j < a.length; j++) { double d = a[j] - b[j]; s += d * d; } return Math.sqrt(s); }
    private static int[] order(double[] F) {
        Integer[] idx = new Integer[F.length]; for (int i = 0; i < F.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (x, y) -> Double.compare(F[x], F[y]));
        int[] o = new int[F.length]; for (int i = 0; i < F.length; i++) o[i] = idx[i]; return o;
    }
}
