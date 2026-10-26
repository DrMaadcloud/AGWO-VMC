package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Glider snake optimizer (El-kenawy et al. [61]). The population is sorted by fitness each iteration; the leader L is the
 * best agent. Agent s (from the tail of the chain to s = 1) moves by Eqs. (1)-(4): S_s + A ((L - S_s) + (S_(s-1) - S_s)),
 * A = 1 - t/T. With replacement probability RP = 0.5, agents in the weaker half are instead re-created by Eq. (5):
 * (r1/N) S_r1 + A (F_l/F_s + F_r2/F_r3), with fitness ratios taken on the shifted fitness (r1 < r2 < r3 random agents).
 * The leader is kept. Implementation follows the published equations and the reference code of the authors as ported in
 * pyMetaheuristic 7.4 (gso_glider_snake).
 */
public final class GsoOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private final double rp = 0.5;
    private LocalMoves lsMoves = null; private boolean useLs = false; private int tLs = 5, lMax = 50;
    public GsoOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) { N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta; }
    public GsoOptimizer withLocalSearch(LocalMoves m, int tLs, int lMax) { useLs = m != null; lsMoves = m; this.tLs = tLs; this.lMax = lMax; return this; }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        SymSpace S = new SymSpace(f, repair, c, beta, maxEvals, rng);
        int n = S.n;
        double[][] Z = new double[N][]; double[] F = new double[N];
        for (int i = 0; i < N; i++) { Z[i] = S.initial(incumbent, i); F[i] = S.eval(Z[i]); }
        S.offerIncumbent(incumbent);
        int T = (int) Math.max(1, (maxEvals - N) / (N - 1));
        List<Double> hist = new ArrayList<>(); hist.add(S.arch.fitness());
        int t = 0;
        while (!S.budget.exhausted()) {
            sortPop(Z, F);
            double A = Math.max(0, 1 - (double) t / T);
            double fmin = F[0];
            double[] ratio = new double[N];
            for (int i = 0; i < N; i++) ratio[i] = F[i] - fmin + 1e-12;
            double[][] Zn = new double[N][]; double[] Fn = F.clone();
            Zn[0] = Z[0];
            for (int s = N - 1; s >= 1 && !S.budget.exhausted(); s--) {
                double mrand = rng.nextDouble();
                int[] r = threeDistinct(rng);
                double[] y = new double[n];
                if (rp > mrand && s > 0.5 * N) {
                    double chain = r[0] / (double) N;
                    double add = A * (ratio[0] / ratio[s] + ratio[r[1]] / ratio[r[2]]);
                    for (int j = 0; j < n; j++) y[j] = chain * Z[r[0]][j] + add;
                } else {
                    for (int j = 0; j < n; j++) y[j] = Z[s][j] + A * ((Z[0][j] - Z[s][j]) + (Z[s - 1][j] - Z[s][j]));
                }
                Fn[s] = S.eval(y); Zn[s] = y;
            }
            for (int s = 1; s < N; s++) if (Zn[s] != null) { Z[s] = Zn[s]; F[s] = Fn[s]; }
            t++;
            if (useLs && t % tLs == 0) S.localSearch(lsMoves, lMax, Z, F);
            hist.add(S.arch.fitness());
            if (log != null) log.onIteration(t, S.arch.fitness(), Double.NaN, Double.NaN, Double.NaN, Diversity.continuous(Z), Double.NaN, Double.NaN);
        }
        return new OptimizerResult(S.arch.best(), S.arch.fitness(), t, S.budget.used(), hist);
    }

    private int[] threeDistinct(Random rng) {
        int a = rng.nextInt(N), b, c3;
        do { b = rng.nextInt(N); } while (b == a);
        do { c3 = rng.nextInt(N); } while (c3 == a || c3 == b);
        int[] r = {a, b, c3}; Arrays.sort(r); return r;
    }
    static void sortPop(double[][] Z, double[] F) {
        Integer[] o = new Integer[F.length]; for (int i = 0; i < o.length; i++) o[i] = i;
        Arrays.sort(o, (x, y) -> Double.compare(F[x], F[y]));
        double[][] Z2 = new double[Z.length][]; double[] F2 = new double[F.length];
        for (int i = 0; i < o.length; i++) { Z2[i] = Z[o[i]]; F2[i] = F[o[i]]; }
        System.arraycopy(Z2, 0, Z, 0, Z.length); System.arraycopy(F2, 0, F, 0, F.length);
    }
}
