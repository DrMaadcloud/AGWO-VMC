package org.agwovmc.revision;

import java.util.Random;

/**
 * Pairwise-swap (2-exchange) local search on the best-so-far placement (Section 4.4).
 * Tries up to lMax random swaps of two VMs hosted on different PMs and keeps a swap only if it
 * strictly lowers F and keeps the placement feasible. Every evaluation is taken from the shared budget.
 */
public final class PairwiseSwapSearch {
    public static void run(ElitistArchive archive, Fitness f, Repair repair, int lMax, EvalBudget budget, Random rng) {
        int[] cur = archive.best();
        if (cur == null) return;
        double fCur = archive.fitness();
        int n = cur.length;
        for (int s = 0; s < lMax && !budget.exhausted(); s++) {
            int i = rng.nextInt(n), j = rng.nextInt(n);
            if (i == j || cur[i] == cur[j]) continue;
            int pi = cur[i];
            cur[i] = cur[j]; cur[j] = pi;
            if (!repair.isFeasible(cur) || !budget.take()) { cur[j] = cur[i]; cur[i] = pi; continue; }
            double fNew = f.evaluate(cur);
            if (fNew < fCur) { fCur = fNew; archive.offer(cur, fNew); }
            else { cur[j] = cur[i]; cur[i] = pi; }
        }
    }
}
