package org.agwovmc.sim;

import java.util.Random;

/**
 * Scenario workloads of Section 6.3, generated from the workload seed (common random numbers:
 * every method sees exactly the same demand for a given scenario and seed).
 * demand[j][c] = CPU demand of VM j in cycle c as a fraction of the VM's MIPS capacity.
 */
public final class Workload {
    public static final double[] VM_MIPS = {1800, 1500, 1000, 500};
    public static final int[] VM_RAM = {870, 1740, 1740, 613};
    public static final int CYCLE_S = 300;

    public final int n, cycles;
    public final double[][] demand;
    public final int[] type;
    public final int spikeCycle;

    public Workload(String scenario, int n, int cycles, long seed) {
        this.n = n; this.cycles = cycles;
        Random r = new Random(seed);
        demand = new double[n][cycles + 1];
        type = new int[n];
        int spike = -1;
        if (scenario.equals("flash")) spike = (int) ((8 * 3600 + r.nextDouble() * 8 * 3600) / CYCLE_S);
        spikeCycle = spike;
        for (int j = 0; j < n; j++) {
            type[j] = r.nextInt(4);
            double mean, phase = r.nextGaussian() * 0.13;
            boolean inGroup = r.nextBoolean();
            int cls = j % 3;
            switch (scenario) {
                case "static": mean = 0.25 + 0.40 * r.nextDouble(); break;
                case "dynamic": mean = 0.20 + 0.40 * r.nextDouble(); break;
                case "flash": mean = 0.15 + 0.30 * r.nextDouble(); break;
                default: // hetero: CPU-intensive analytics, memory-intensive caching, balanced web
                    if (cls == 0) { mean = 0.60 + 0.25 * r.nextDouble(); type[j] = r.nextBoolean() ? 0 : 3; }
                    else if (cls == 1) { mean = 0.10 + 0.20 * r.nextDouble(); type[j] = r.nextBoolean() ? 1 : 2; }
                    else { mean = 0.30 + 0.20 * r.nextDouble(); }
            }
            for (int c = 0; c <= cycles; c++) {
                double t = (double) c * CYCLE_S / 86400.0, d;
                switch (scenario) {
                    case "static": d = mean + r.nextGaussian() * 0.05; break;
                    case "dynamic": d = mean * (1 + 0.3 * Math.sin(2 * Math.PI * (t - 0.25) + phase)) + r.nextGaussian() * 0.03; break;
                    case "flash":
                        d = mean + r.nextGaussian() * 0.04;
                        if (inGroup && spike >= 0 && c >= spike && c < spike + 3) d *= 2.0;   // 2x for 15 minutes
                        break;
                    default:
                        d = (cls == 2) ? mean * (1 + 0.3 * Math.sin(2 * Math.PI * (t - 0.25) + phase)) + r.nextGaussian() * 0.03
                                       : mean + r.nextGaussian() * 0.04;
                }
                demand[j][c] = Math.min(1.0, Math.max(0.01, d));
            }
        }
    }

    public double at(int j, double timeSeconds) {
        int c = (int) Math.min(cycles, Math.max(0, Math.floor(timeSeconds / CYCLE_S)));
        return demand[j][c];
    }
}
