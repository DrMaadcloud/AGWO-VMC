package org.agwovmc.revision;

/**
 * Fitness of a VM-to-PM placement for ONE consolidation cycle (Eq. 8 of the revised manuscript).
 * Implement this on top of your CloudSim Plus model. The demand snapshot and the normalisation
 * constants must stay fixed while one optimisation run is in progress (Assumption A1).
 */
public interface Fitness {
    /** Weighted fitness F(phi) (lower is better). placement[j] = PM index (0-based) of VM j. */
    double evaluate(int[] placement);

    /** The four normalised objective terms (E_norm, SLA_viol_norm, E_mig_norm, U_imbalance_norm). */
    double[] objectives(int[] placement);

    /** The four objective terms followed by the relative capacity excess (0 for a feasible placement), from one pass. */
    default double[] objectivesAndViolation(int[] placement) {
        double[] o = objectives(placement);
        return new double[]{o[0], o[1], o[2], o[3], 0.0};
    }

    int numVms();

    int numPms();
}
