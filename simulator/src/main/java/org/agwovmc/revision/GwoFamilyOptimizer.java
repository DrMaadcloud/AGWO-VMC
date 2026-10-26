package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * AGWO-VMC and the GWO-family baselines on a shared encoding, repair operator and evaluation budget.
 *  - AGWO-VMC:     decay=POWER, gamma=0.2, lambda=1.9, k=5, beta=1.5, localSearch=true
 *  - standard GWO: decay=LINEAR, gamma=0, localSearch=false (beta=1.5 in the main comparison,
 *                  beta<=0 = hard rounding for the "no adaptations" ablation row)
 *  - mGWO [30]:    decay=MGWO
 *  - chaotic GWO:  chaotic=true (logistic map replaces the uniform r1, r2; check against [27])
 *  - ablation:     any combination of decay / gamma / beta / localSearch
 */
public final class GwoFamilyOptimizer {
    public static final class Config {
        public int populationSize = 50;
        public long maxEvaluations = 12_000;
        public ImprovementFeedback.Decay decay = ImprovementFeedback.Decay.POWER;
        public double lambda = 1.9, gamma = 0.2, beta = 1.5;
        public int k = 5, candidateSize = 32, tLs = 5, lMax = 50;
        public boolean localSearch = true, chaotic = false;
        /** Optional host-evacuation moves; half of the local-search budget L_max is spent on them. */
        public LocalMoves evacuation = null;
    }

    private final Config cfg;
    public GwoFamilyOptimizer(Config cfg) { this.cfg = cfg; }

    /** Planned iteration count so that the schedule ends when the evaluation budget does. */
    public int plannedIterations() {
        double perIter = cfg.populationSize + (cfg.localSearch ? (double) cfg.lMax / cfg.tLs : 0.0);
        return (int) Math.max(1, Math.floor((cfg.maxEvaluations - cfg.populationSize) / perIter));
    }

    public OptimizerResult optimise(Fitness f, Repair repair, int[] incumbent, Random rng, IterationListener log) {
        int n = f.numVms(), m = f.numPms(), N = cfg.populationSize;
        int tMax = plannedIterations();
        EvalBudget budget = new EvalBudget(cfg.maxEvaluations);
        CandidateSoftmaxDecoder dec = new CandidateSoftmaxDecoder(m, cfg.candidateSize, cfg.beta);
        ImprovementFeedback fb = new ImprovementFeedback(cfg.decay, cfg.lambda, cfg.gamma, cfg.k, tMax);
        ElitistArchive archive = new ElitistArchive();
        double[][] X = new double[N][n];
        int[][] P = new int[N][];
        double[] F = new double[N];
        double chaos = 0.7;

        // initialisation: wolf 0 starts from the incumbent placement (if given), the rest uniformly at random
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < n; j++) X[i][j] = Init.coord(incumbent, i, j, m, rng);
            P[i] = dec.decode(X[i], rng); repair.repair(P[i]);
            if (!budget.take()) break;
            F[i] = f.evaluate(P[i]);
            archive.offer(P[i], F[i]);
        }
        if (incumbent != null && budget.take()) { int[] inc = incumbent.clone(); repair.repair(inc); archive.offer(inc, f.evaluate(inc)); }
        List<Double> hist = new ArrayList<>();
        fb.record(archive.fitness()); hist.add(archive.fitness());
        int t = 0;
        while (!budget.exhausted() && t < tMax) {
            t++;
            double a = fb.coefficient(t);
            // leaders: alpha = best-so-far archive, beta/delta = two best current wolves
            int b1 = -1, b2 = -1;
            for (int i = 0; i < N; i++) {
                if (b1 < 0 || F[i] < F[b1]) { b2 = b1; b1 = i; }
                else if (b2 < 0 || F[i] < F[b2]) b2 = i;
            }
            int[] alphaP = archive.best();
            double[][] L = new double[3][n];
            for (int j = 0; j < n; j++) { L[0][j] = alphaP[j]; L[1][j] = X[b1][j]; L[2][j] = X[b2][j]; }
            int repaired = 0, evaluated = 0;
            for (int i = 0; i < N && !budget.exhausted(); i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int l = 0; l < 3; l++) {
                        double r1, r2;
                        if (cfg.chaotic) { chaos = 4 * chaos * (1 - chaos); r1 = chaos; chaos = 4 * chaos * (1 - chaos); r2 = chaos; }
                        else { r1 = rng.nextDouble(); r2 = rng.nextDouble(); }
                        double A = 2 * a * r1 - a, C = 2 * r2;
                        double D = Math.abs(C * L[l][j] - X[i][j]);
                        sum += L[l][j] - A * D;
                    }
                    X[i][j] = Math.min(m - 1, Math.max(0, sum / 3.0));
                }
                P[i] = dec.decode(X[i], rng);
                if (repair.repair(P[i])) repaired++;
                if (!budget.take()) break;
                F[i] = f.evaluate(P[i]); evaluated++;
                archive.offer(P[i], F[i]);
            }
            if (cfg.localSearch && t % cfg.tLs == 0) LocalSearch.apply(archive, f, repair, cfg.lMax, cfg.evacuation, budget, rng);
            fb.record(archive.fitness()); hist.add(archive.fitness());
            if (log != null)
                log.onIteration(t, archive.fitness(), a, fb.lastG(), fb.lastGbar(),
                        Diversity.continuous(X), Diversity.placement(P), evaluated == 0 ? 0 : (double) repaired / evaluated);
        }
        return new OptimizerResult(archive.best(), archive.fitness(), t, budget.used(), hist);
    }
}
