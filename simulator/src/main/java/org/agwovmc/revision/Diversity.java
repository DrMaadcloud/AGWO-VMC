package org.agwovmc.revision;

import java.util.Arrays;

/** Diversity measures of Section 7.5. */
public final class Diversity {
    /** Dimension-wise diversity of Hussain et al. [56]: mean over dimensions of mean |median(x_j) - x_ij|. */
    public static double continuous(double[][] pop) {
        int nAgents = pop.length, d = pop[0].length;
        double[] col = new double[nAgents];
        double total = 0;
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < nAgents; i++) col[i] = pop[i][j];
            double[] s = col.clone(); Arrays.sort(s);
            double med = (nAgents % 2 == 1) ? s[nAgents / 2] : 0.5 * (s[nAgents / 2 - 1] + s[nAgents / 2]);
            double acc = 0;
            for (int i = 0; i < nAgents; i++) acc += Math.abs(med - col[i]);
            total += acc / nAgents;
        }
        return total / d;
    }

    /** Placement diversity: mean over all pairs of decoded wolves of the fraction of VMs on different PMs. */
    public static double placement(int[][] decoded) {
        int nAgents = decoded.length, n = decoded[0].length;
        long pairs = 0; double acc = 0;
        for (int a = 0; a < nAgents; a++)
            for (int b = a + 1; b < nAgents; b++) {
                int diff = 0;
                for (int j = 0; j < n; j++) if (decoded[a][j] != decoded[b][j]) diff++;
                acc += (double) diff / n; pairs++;
            }
        return pairs == 0 ? 0 : acc / pairs;
    }
}
