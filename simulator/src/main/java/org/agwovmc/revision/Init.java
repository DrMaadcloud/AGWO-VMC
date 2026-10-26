package org.agwovmc.revision;

import java.util.Random;

/**
 * Population initialisation shared by every population method: agent 0 is the current placement; the others
 * start from it with 5-20% of the VMs reassigned uniformly at random (warm start from the incumbent).
 */
public final class Init {
    public static double coord(int[] incumbent, int agent, int j, int m, Random rng) {
        if (incumbent == null) return rng.nextDouble() * (m - 1);
        if (agent == 0) return incumbent[j];
        double q = 0.05 * (1 + agent % 4);
        return rng.nextDouble() < q ? rng.nextDouble() * (m - 1) : incumbent[j];
    }
}
