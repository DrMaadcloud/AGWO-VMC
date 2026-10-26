package org.agwovmc.revision;

/** Best-so-far placement; replaced only by a strictly better one (Assumption A2, Lemma 1). */
public final class ElitistArchive {
    private int[] best;
    private double fBest = Double.POSITIVE_INFINITY;

    public boolean offer(int[] placement, double f) {
        if (f < fBest) { best = placement.clone(); fBest = f; return true; }
        return false;
    }

    public int[] best() { return best == null ? null : best.clone(); }
    public double fitness() { return fBest; }
}
