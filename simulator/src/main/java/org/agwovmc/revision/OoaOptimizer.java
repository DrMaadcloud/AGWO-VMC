package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Octopus optimization algorithm (Wang et al. [62]) with the recoil mechanism.
 * Hunting (exploration): every agent keeps a velocity V and a personal best. With a = 2(1 - t/T) and A = 2 a r - a,
 * |A| > 1 selects the searching-for-prey or the predator-escape move (equal probability), |A| <= 1 the prey-attack move:
 *   V' = F V + sum_k A_k exp(-beta_k sqrt(r_k) g_k(t)) (X_k - X), X' = X + V' + |A| (V'^2 - V^2) / (2 al)  (recoil),
 * with fluid resistance F = w/2 (1 - t/T)^(1 - exp(-t/T)), strengths A_1 = r 2^u, A_2 = r 2^mu, A_3 = r 2^phi,
 * recoil acceleration al = al0 (t/T)^(1 - sqrt(t/T)); perception decays exp(-beta sqrt(r) t/T) towards random agents and
 * exp(-beta sqrt(r) (1 - sqrt(t/T))) / exp(-beta sqrt(r) (1 - t/T)) towards the personal / global best; r_k are
 * root-mean-square distances. Mating (exploitation, greedy): with a random partner at RMS distance R, arms grasping
 * (R <= pl), arms extending (pl < R <= 2 pl) or hectocotylus severing (R > 2 pl).
 * Parameters: w = 0.5, u = mu = phi = 0.5, beta_1..3 = 1, al0 = 5000, pl = 0.5; velocities are bounded by the half-range.
 * The velocity bound and the RMS normalisation of distances are not specified in [62] and were set here.
 */
public final class OoaOptimizer {
    private final int N; private final long maxEvals; private final int c; private final double beta;
    private final double w = 0.5, u = 0.5, mu = 0.5, phi = 0.5, b1 = 1, b2 = 1, b3 = 1, al0 = 5000, pl = 0.5, vmax = 1.0;
    private LocalMoves lsMoves = null; private boolean useLs = false; private int tLs = 5, lMax = 50;
    public OoaOptimizer(int populationSize, long maxEvaluations, int candidateSize, double beta) { N = populationSize; maxEvals = maxEvaluations; c = candidateSize; this.beta = beta; }
    public OoaOptimizer withLocalSearch(LocalMoves m, int tLs, int lMax) { useLs = m != null; lsMoves = m; this.tLs = tLs; this.lMax = lMax; return this; }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        SymSpace S = new SymSpace(f, repair, c, beta, maxEvals, rng);
        int n = S.n;
        double[][] Z = new double[N][], V = new double[N][n], PB = new double[N][]; double[] F = new double[N], FPB = new double[N];
        for (int i = 0; i < N; i++) { Z[i] = S.initial(incumbent, i); F[i] = S.eval(Z[i]); PB[i] = Z[i].clone(); FPB[i] = F[i]; }
        S.offerIncumbent(incumbent);
        int T = (int) Math.max(1, (maxEvals - N) / (2 * N));
        List<Double> hist = new ArrayList<>(); hist.add(S.arch.fitness());
        int t = 0;
        while (!S.budget.exhausted()) {
            t++;
            double tau = Math.min(1.0, (double) t / T);
            double a = 2 * (1 - tau);
            double Fr = 0.5 * w * Math.pow(1 - tau, 1 - Math.exp(-tau));
            double al = Math.max(1e-9, al0 * Math.pow(tau, 1 - Math.sqrt(tau)));
            double[] gb = S.toZ(S.arch.best());
            for (int i = 0; i < N && !S.budget.exhausted(); i++) {
                double A = 2 * a * rng.nextDouble() - a;
                double[] X = Z[i], Vn = new double[n];
                if (Math.abs(A) > 1) {
                    if (rng.nextBoolean()) {     // searching for prey: random memory position and personal best
                        double[] k0 = Z[rng.nextInt(N)];
                        double g0 = rng.nextDouble() * Math.pow(2, u) * Math.exp(-b2 * Math.sqrt(rms(k0, X)) * tau);
                        double g1 = rng.nextDouble() * Math.pow(2, phi) * Math.exp(-b1 * Math.sqrt(rms(PB[i], X)) * (1 - Math.sqrt(tau)));
                        for (int j = 0; j < n; j++) Vn[j] = Fr * V[i][j] + g0 * (k0[j] - X[j]) + g1 * (PB[i][j] - X[j]);
                    } else {                     // escaping predators along two directions
                        double[] k1 = Z[rng.nextInt(N)], k2 = Z[rng.nextInt(N)];
                        double g1 = rng.nextDouble() * Math.pow(2, u) * Math.exp(-b2 * Math.sqrt(rms(k1, X)) * tau);
                        double g2 = rng.nextDouble() * Math.pow(2, u) * Math.exp(-b2 * Math.sqrt(rms(k2, X)) * tau);
                        for (int j = 0; j < n; j++) Vn[j] = Fr * V[i][j] + g1 * (k1[j] - X[j]) + g2 * (k2[j] - X[j]);
                    }
                } else {                         // attacking prey: personal and global best
                    double g1 = rng.nextDouble() * Math.pow(2, phi) * Math.exp(-b1 * Math.sqrt(rms(PB[i], X)) * (1 - Math.sqrt(tau)));
                    double g2 = rng.nextDouble() * Math.pow(2, mu) * Math.exp(-b3 * Math.sqrt(rms(gb, X)) * (1 - tau));
                    for (int j = 0; j < n; j++) Vn[j] = Fr * V[i][j] + g1 * (PB[i][j] - X[j]) + g2 * (gb[j] - X[j]);
                }
                double[] y = new double[n];
                for (int j = 0; j < n; j++) {
                    Vn[j] = Math.max(-vmax, Math.min(vmax, Vn[j]));
                    y[j] = X[j] + Vn[j] + Math.abs(A) * (Vn[j] * Vn[j] - V[i][j] * V[i][j]) / (2 * al);
                }
                S.clamp(y);
                double fy = S.eval(y);
                Z[i] = y; F[i] = fy; V[i] = Vn;
                if (fy < FPB[i]) { PB[i] = y.clone(); FPB[i] = fy; }
                if (S.budget.exhausted()) break;
                // mating
                int k = rng.nextInt(N - 1); if (k >= i) k++;
                double[] male = Z[i], fem = Z[k], ch = new double[n];
                double R = rms(male, fem), p = rng.nextDouble();
                if (R <= pl) for (int j = 0; j < n; j++) ch[j] = p * male[j] + (1 - p) * fem[j];
                else if (R <= 2 * pl) for (int j = 0; j < n; j++) ch[j] = male[j] + p * (fem[j] - male[j]);
                else for (int j = 0; j < n; j++) ch[j] = fem[j] + p * (male[j] - fem[j]);
                double fc = S.eval(ch);
                if (fc < F[i]) { Z[i] = ch; F[i] = fc; if (fc < FPB[i]) { PB[i] = ch.clone(); FPB[i] = fc; } }
            }
            if (useLs && t % tLs == 0) {
                S.localSearch(lsMoves, lMax, Z, F);
                for (int i = 0; i < N; i++) if (F[i] < FPB[i]) { PB[i] = Z[i].clone(); FPB[i] = F[i]; }
            }
            hist.add(S.arch.fitness());
            if (log != null) log.onIteration(t, S.arch.fitness(), a, Double.NaN, Double.NaN, Diversity.continuous(Z), Double.NaN, Double.NaN);
        }
        return new OptimizerResult(S.arch.best(), S.arch.fitness(), t, S.budget.used(), hist);
    }

    private static double rms(double[] a, double[] b) {
        double s = 0; for (int j = 0; j < a.length; j++) { double d = a[j] - b[j]; s += d * d; }
        return Math.sqrt(s / a.length);
    }
}
