package org.agwovmc.revision;

import java.util.List;

public final class OptimizerResult {
    public final int[] placement;
    public final double fitness;
    public final int iterations;
    public final long evaluations;
    public final int convIter95;
    public final List<Double> bestHistory;

    OptimizerResult(int[] p, double f, int it, long ev, List<Double> hist) {
        placement = p; fitness = f; iterations = it; evaluations = ev; bestHistory = hist;
        convIter95 = conv95(hist);
    }

    /** First iteration at which 95% of the total improvement (initial -> final) has been achieved. */
    static int conv95(List<Double> h) {
        if (h.isEmpty()) return 0;
        double f0 = h.get(0), fe = h.get(h.size() - 1), target = f0 - 0.95 * (f0 - fe);
        if (f0 - fe <= 0) return 0;
        for (int i = 0; i < h.size(); i++) if (h.get(i) <= target + 1e-15) return i;
        return h.size() - 1;
    }
}
