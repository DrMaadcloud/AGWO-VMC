package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Harris hawks optimisation [26] on the shared encoding, repair and budget. */
public final class HhoOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private LocalMoves lsMoves = null; private int tLs = 5, lMax = 50;
    /** Adds the AGWO-VMC local search (swaps + host evacuation) every tLs iterations, for the like-for-like "+LS" baseline. */
    public HhoOptimizer withLocalSearch(LocalMoves evacuation, int tLs, int lMax) { this.lsMoves = evacuation; this.tLs = tLs; this.lMax = lMax; return this; }
    public HhoOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) {
        N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta;
    }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        int n = f.numVms(), m = f.numPms();
        EvalBudget budget = new EvalBudget(maxEvals);
        CandidateSoftmaxDecoder dec = new CandidateSoftmaxDecoder(m, c, beta);
        ElitistArchive arch = new ElitistArchive();
        double[][] X = new double[N][n]; double[] F = new double[N];
        int T = (int) Math.max(1, (maxEvals - N) / (1.5 * N));   // average cost per hawk is about 1.5 evaluations
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < n; j++) X[i][j] = Init.coord(incumbent, i, j, m, rng);
            F[i] = eval(X[i], f, repair, dec, budget, arch, rng);
        }
        if (incumbent != null && budget.take()) { int[] inc = incumbent.clone(); repair.repair(inc); arch.offer(inc, f.evaluate(inc)); }
        List<Double> hist = new ArrayList<>(); hist.add(arch.fitness());
        double[] rabbit = toD(arch.best()); int t = 0;
        while (!budget.exhausted()) {
            t++;
            double[] mean = new double[n];
            for (double[] x : X) for (int j = 0; j < n; j++) mean[j] += x[j] / N;
            for (int i = 0; i < N && !budget.exhausted(); i++) {
                double E0 = 2 * rng.nextDouble() - 1, E = 2 * E0 * (1 - (double) t / T), J = 2 * (1 - rng.nextDouble());
                double[] y = new double[n];
                if (Math.abs(E) >= 1) {
                    if (rng.nextDouble() >= 0.5) {
                        double[] xr = X[rng.nextInt(N)];
                        double r1 = rng.nextDouble(), r2 = rng.nextDouble();
                        for (int j = 0; j < n; j++) y[j] = xr[j] - r1 * Math.abs(xr[j] - 2 * r2 * X[i][j]);
                    } else {
                        double r3 = rng.nextDouble(), r4 = rng.nextDouble();
                        for (int j = 0; j < n; j++) y[j] = (rabbit[j] - mean[j]) - r3 * (r4 * (m - 1));
                    }
                    clamp(y, m); F[i] = eval(y, f, repair, dec, budget, arch, rng); X[i] = y;
                } else {
                    double r = rng.nextDouble();
                    if (r >= 0.5 && Math.abs(E) >= 0.5) {
                        for (int j = 0; j < n; j++) y[j] = (rabbit[j] - X[i][j]) - E * Math.abs(J * rabbit[j] - X[i][j]);
                        clamp(y, m); F[i] = eval(y, f, repair, dec, budget, arch, rng); X[i] = y;
                    } else if (r >= 0.5) {
                        for (int j = 0; j < n; j++) y[j] = rabbit[j] - E * Math.abs(rabbit[j] - X[i][j]);
                        clamp(y, m); F[i] = eval(y, f, repair, dec, budget, arch, rng); X[i] = y;
                    } else {
                        double[] base = (Math.abs(E) >= 0.5) ? X[i] : mean;
                        for (int j = 0; j < n; j++) y[j] = rabbit[j] - E * Math.abs(J * rabbit[j] - base[j]);
                        clamp(y, m);
                        double fy = eval(y, f, repair, dec, budget, arch, rng);
                        if (fy < F[i]) { X[i] = y; F[i] = fy; }
                        else if (!budget.exhausted()) {
                            double[] z = new double[n];
                            for (int j = 0; j < n; j++) z[j] = y[j] + rng.nextDouble() * levy(rng) * (m - 1) * 0.01;
                            clamp(z, m);
                            double fz = eval(z, f, repair, dec, budget, arch, rng);
                            if (fz < F[i]) { X[i] = z; F[i] = fz; }
                        }
                    }
                }
            }
            if (lsMoves != null && t % tLs == 0) LocalSearch.apply(arch, f, repair, lMax, lsMoves, budget, rng);
            rabbit = toD(arch.best()); hist.add(arch.fitness());
            if (log != null) log.onIteration(t, arch.fitness(), Double.NaN, Double.NaN, Double.NaN, Diversity.continuous(X), Double.NaN, Double.NaN);
        }
        return new OptimizerResult(arch.best(), arch.fitness(), t, budget.used(), hist);
    }

    private static double levy(Random rng) {
        double b = 1.5;
        double sigma = Math.pow(gamma(1 + b) * Math.sin(Math.PI * b / 2) / (gamma((1 + b) / 2) * b * Math.pow(2, (b - 1) / 2)), 1 / b);
        return rng.nextGaussian() * sigma / Math.pow(Math.abs(rng.nextGaussian()), 1 / b);
    }
    private static double gamma(double x) { // Lanczos approximation
        double[] g = {676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
                12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
        if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gamma(1 - x));
        x -= 1; double a = 0.99999999999980993; double t = x + 7.5;
        for (int i = 0; i < 8; i++) a += g[i] / (x + i + 1);
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
    }
    static void clamp(double[] y, int m) { for (int j = 0; j < y.length; j++) y[j] = Math.min(m - 1, Math.max(0, y[j])); }
    static double[] toD(int[] p) { double[] d = new double[p.length]; for (int j = 0; j < p.length; j++) d[j] = p[j]; return d; }
    static double eval(double[] x, Fitness f, Repair rep, CandidateSoftmaxDecoder dec, EvalBudget b, ElitistArchive arch, Random rng) {
        if (b.exhausted()) return Double.POSITIVE_INFINITY;
        int[] p = dec.decode(x, rng); rep.repair(p); b.take();
        double v = f.evaluate(p); arch.offer(p, v); return v;
    }
}
