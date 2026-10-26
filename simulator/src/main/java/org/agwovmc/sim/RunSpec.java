package org.agwovmc.sim;

import java.util.LinkedHashMap;
import java.util.Map;

/** One simulation run of the campaign. */
public final class RunSpec {
    public final String experiment, scenario, method, config;
    public final int seed, fleet, cycles;
    public final Map<String, String> params = new LinkedHashMap<>();
    public double[] weights = {0.50, 0.30, 0.15, 0.05};
    public boolean logIterations = false, logFronts = false;

    public RunSpec(String experiment, String scenario, String method, String config, int seed, int fleet, int cycles) {
        this.experiment = experiment; this.scenario = scenario; this.method = method; this.config = config;
        this.seed = seed; this.fleet = fleet; this.cycles = cycles;
    }
    public RunSpec p(String k, Object v) { params.put(k, String.valueOf(v)); return this; }
    public String id() { return experiment + "__" + scenario + "__" + method + "__" + config + "__" + fleet + "__s" + (seed < 10 ? "0" : "") + seed; }
    public String param(String k, String d) { return params.getOrDefault(k, d); }
}
