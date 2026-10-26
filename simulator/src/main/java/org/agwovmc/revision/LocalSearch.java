package org.agwovmc.revision;

import java.util.Random;

/**
 * Local search applied to the best-so-far placement every T_ls iterations (Section 4.4): L_max/2 pairwise swaps
 * followed by L_max/2 host-evacuation proposals when an evacuation move generator is available (otherwise L_max swaps).
 * Every evaluation is charged to the shared budget. Used by AGWO-VMC and, for the like-for-like comparison, by the
 * "+LS" variants of GA, PSO, HHO, I-GWO and the GWO variants.
 */
public final class LocalSearch {
    private LocalSearch() { }
    public static void apply(ElitistArchive archive, Fitness f, Repair repair, int lMax, LocalMoves evacuation, EvalBudget budget, Random rng) {
        int swaps = evacuation == null ? lMax : lMax / 2;
        PairwiseSwapSearch.run(archive, f, repair, swaps, budget, rng);
        for (int e = 0; evacuation != null && e < lMax - swaps && !budget.exhausted(); e++) {
            int[] q = evacuation.propose(archive.best(), rng);
            if (q == null || !budget.take()) continue;
            archive.offer(q, f.evaluate(q));
        }
    }
}
