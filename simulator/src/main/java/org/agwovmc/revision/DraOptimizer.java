package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Divine religions algorithm (Mozhdehi et al. [63]), following the authors' MATLAB reference code as ported in
 * pyMetaheuristic 7.4 (dra): belief-profile consideration for a new follower (BPSP = 0.5), miracle operator with rate
 * MP = r (1 - 2t/T) r, proselytism towards the leader or around the follower mean, reward/penalty on a random follower
 * (rate 0.2) and replacement of the worst belief by the new follower when it is better. Greedy acceptance.
 */
public final class DraOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private final double bpsp = 0.5, rpr = 0.2;
    private LocalMoves lsMoves = null; private boolean useLs = false; private int tLs = 5, lMax = 50;
    public DraOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) { N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta; }
    public DraOptimizer withLocalSearch(LocalMoves m, int tLs, int lMax) { useLs = m != null; lsMoves = m; this.tLs = tLs; this.lMax = lMax; return this; }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        SymSpace S = new SymSpace(f, repair, c, beta, maxEvals, rng);
        int n = S.n;
        double[][] Z = new double[N][]; double[] F = new double[N];
        for (int i = 0; i < N; i++) { Z[i] = S.initial(incumbent, i); F[i] = S.eval(Z[i]); }
        S.offerIncumbent(incumbent);
        int T = (int) Math.max(1, (maxEvals - N) / (N + 2));
        List<Double> hist = new ArrayList<>(); hist.add(S.arch.fitness());
        int t = 0;
        while (!S.budget.exhausted()) {
            t++;
            double MP = rng.nextDouble() * (1.0 - ((double) t / T * 2.0)) * rng.nextDouble();
            GsoOptimizer.sortPop(Z, F);
            double[] leader = Z[0].clone();
            // new follower from the belief profile
            double[] nf = new double[n];
            for (int j = 0; j < n; j++) nf[j] = SymSpace.LB + rng.nextDouble() * (SymSpace.UB - SymSpace.LB);
            if (rng.nextDouble() <= bpsp) nf[rng.nextInt(n)] = Z[rng.nextInt(N)][rng.nextInt(n)];
            double fnf = S.eval(nf);
            for (int i = 0; i < N && !S.budget.exhausted(); i++) {
                double[] cur = Z[i], y = new double[n];
                if (rng.nextDouble() <= MP) {                     // miracle operator
                    if (rng.nextDouble() <= 0.5) {
                        double k = Math.cos(Math.PI / 2) * (rng.nextDouble() - Math.cos(rng.nextDouble()));
                        for (int j = 0; j < n; j++) y[j] = cur[j] * k;
                    } else {
                        double r = rng.nextDouble(), k = Math.round(Math.pow(1.0, rng.nextDouble()));
                        for (int j = 0; j < n; j++) y[j] = cur[j] + r * (cur[j] - k * cur[j]);
                    }
                } else if (rng.nextDouble() > 1.0 - MP) {         // proselytism around the follower mean
                    double mean = 0; for (double v : cur) mean += v; mean /= n;
                    double add = mean * (1.0 - MP) + (1.0 - mean) - (rng.nextDouble() - 4.0 * Math.sin(Math.sin(Math.PI * rng.nextDouble())));
                    for (int j = 0; j < n; j++) y[j] = cur[j] * 0.01 + add;
                } else {                                          // proselytism towards the leader
                    double k = rng.nextDouble() - Math.cos(rng.nextDouble());
                    for (int j = 0; j < n; j++) y[j] = leader[j] * k;
                }
                double fy = S.eval(y);
                if (fy < F[i]) { Z[i] = y; F[i] = fy; }
            }
            int w = 0; for (int i = 1; i < N; i++) if (F[i] > F[w]) w = i;
            if (fnf < F[w]) { Z[w] = nf; F[w] = fnf; }
            // reward / penalty
            int idx = rng.nextInt(N);
            double scale = rng.nextDouble() >= rpr ? 1.0 - rng.nextGaussian() : 1.0 + rng.nextGaussian();
            double[] y = new double[n]; for (int j = 0; j < n; j++) y[j] = Z[idx][j] * scale;
            double fy = S.eval(y);
            if (fy < F[idx]) { Z[idx] = y; F[idx] = fy; }
            if (useLs && t % tLs == 0) S.localSearch(lsMoves, lMax, Z, F);
            hist.add(S.arch.fitness());
            if (log != null) log.onIteration(t, S.arch.fitness(), MP, Double.NaN, Double.NaN, Diversity.continuous(Z), Double.NaN, Double.NaN);
        }
        return new OptimizerResult(S.arch.best(), S.arch.fitness(), t, S.budget.used(), hist);
    }
}
