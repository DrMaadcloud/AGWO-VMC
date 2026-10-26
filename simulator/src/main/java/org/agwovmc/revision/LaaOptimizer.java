package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Levy arithmetic algorithm (Barua and Merabet [60]): the arithmetic optimisation algorithm with a Levy random step S
 * (Mantegna, index 1.5) in the arithmetic operators. Math optimiser accelerator MOA(t) = 0.2 + t (1 - 0.2)/T; math
 * optimiser probability MOP(t) = 1 - t^(1/alpha) / T^(1/alpha), alpha = 5; control parameter mu = 0.499. Per dimension:
 * if r1 > MOA (exploration) division (r2 > 0.5) or multiplication of the best solution, otherwise (exploitation)
 * subtraction (r3 > 0.5) or addition, each with the step ((UB - LB) mu S + LB). Greedy replacement.
 */
public final class LaaOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private final double alpha = 5, mu = 0.499, lev = 1.5;
    private LocalMoves lsMoves = null; private boolean useLs = false; private int tLs = 5, lMax = 50;
    public LaaOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) { N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta; }
    public LaaOptimizer withLocalSearch(LocalMoves m, int tLs, int lMax) { useLs = m != null; lsMoves = m; this.tLs = tLs; this.lMax = lMax; return this; }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        SymSpace S = new SymSpace(f, repair, c, beta, maxEvals, rng);
        int n = S.n;
        double[][] Z = new double[N][]; double[] F = new double[N];
        for (int i = 0; i < N; i++) { Z[i] = S.initial(incumbent, i); F[i] = S.eval(Z[i]); }
        S.offerIncumbent(incumbent);
        int T = (int) Math.max(1, (maxEvals - N) / N);
        List<Double> hist = new ArrayList<>(); hist.add(S.arch.fitness());
        int t = 0; final double eps = 2.2204e-16, rangeTerm = SymSpace.UB - SymSpace.LB;
        while (!S.budget.exhausted()) {
            t++;
            double moa = 0.2 + t * (1 - 0.2) / T;
            double mop = 1 - Math.pow(Math.min(t, T), 1 / alpha) / Math.pow(T, 1 / alpha);
            double[] best = S.toZ(S.arch.best());
            for (int i = 0; i < N && !S.budget.exhausted(); i++) {
                double[] y = new double[n];
                for (int j = 0; j < n; j++) {
                    double step = rangeTerm * mu * S.levy(lev) + SymSpace.LB;
                    if (rng.nextDouble() > moa) y[j] = rng.nextDouble() > 0.5 ? best[j] / (mop + eps) * step : best[j] * mop * step;
                    else y[j] = rng.nextDouble() > 0.5 ? best[j] - mop * step : best[j] + mop * step;
                }
                double fy = S.eval(y);
                if (fy < F[i]) { Z[i] = y; F[i] = fy; }
            }
            if (useLs && t % tLs == 0) S.localSearch(lsMoves, lMax, Z, F);
            hist.add(S.arch.fitness());
            if (log != null) log.onIteration(t, S.arch.fitness(), Double.NaN, Double.NaN, Double.NaN, Diversity.continuous(Z), Double.NaN, Double.NaN);
        }
        return new OptimizerResult(S.arch.best(), S.arch.fitness(), t, S.budget.used(), hist);
    }
}
