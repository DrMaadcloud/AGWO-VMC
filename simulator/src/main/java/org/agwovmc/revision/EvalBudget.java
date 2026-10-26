package org.agwovmc.revision;

/** Counts fitness evaluations so that every population method gets the same budget (12,000 per cycle). */
public final class EvalBudget {
    private final long max;
    private long used;

    public EvalBudget(long max) { this.max = max; }

    /** Returns false (and does not count) when the budget is exhausted. */
    public boolean take() {
        if (used >= max) return false;
        used++;
        return true;
    }

    public boolean exhausted() { return used >= max; }
    public long used() { return used; }
    public long max() { return max; }
}
