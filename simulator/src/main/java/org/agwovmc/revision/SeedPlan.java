package org.agwovmc.revision;

/**
 * Common random numbers (Section 6.5): seed s fixes the workload realisation for a scenario,
 * identically for every method; each method gets its own algorithm stream derived from s.
 */
public final class SeedPlan {
    public static long workloadSeed(int seed, String scenario) { return mix(seed * 1_000_003L + scenario.hashCode()); }
    public static long algorithmSeed(int seed, String scenario, String method) { return mix(workloadSeed(seed, scenario) ^ (method.hashCode() * 0x9E3779B97F4A7C15L)); }
    static long mix(long z) { // SplitMix64 finaliser
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
