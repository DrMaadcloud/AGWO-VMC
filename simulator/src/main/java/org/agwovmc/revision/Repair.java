package org.agwovmc.revision;

/** Greedy capacity repair: move each VM that violates a capacity dimension to the least-loaded feasible PM. */
public interface Repair {
    /** Repairs the placement in place. Returns true if anything was changed. */
    boolean repair(int[] placement);

    boolean isFeasible(int[] placement);
}
