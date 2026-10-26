package org.agwovmc.sim;

/** Eq. (1): P(u) = P_idle + (P_max - P_idle)(2u - u^1.4), with SPECpower idle/peak values of the ML110 G4 and G5. */
public final class Power {
    public static final double[] IDLE = {86.0, 93.7}, MAX = {117.0, 135.0}, MIPS_PER_PE = {1860, 2660};
    public static double watts(int hostType, double u) {
        u = Math.min(1.0, Math.max(0.0, u));
        return IDLE[hostType] + (MAX[hostType] - IDLE[hostType]) * (2 * u - Math.pow(u, 1.4));
    }
}
