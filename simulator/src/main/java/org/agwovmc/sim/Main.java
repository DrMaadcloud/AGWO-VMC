package org.agwovmc.sim;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Reduced-scale revision campaign for AGWO-VMC in CloudSim Plus.
 * Writes results/runs.csv (one line per finished run; the campaign resumes where it stopped),
 * results/iters/*.csv (iteration logs), results/progress.txt and results/DONE.txt at the end.
 * Optional program argument: a comma-separated list of experiments to run (default: all).
 */
public final class Main {
    static final String[] SCEN = {"static", "dynamic", "flash", "hetero"};
    static final int FLEET = 100, CYCLES = 288;

    public static void main(String[] args) throws Exception {
        org.cloudsimplus.util.Log.setLevel(ch.qos.logback.classic.Level.OFF);
        // Default: the v3 campaign (NSGA-II and the four recent metaheuristics) in results_v3.
        // First argument "v2" re-runs / resumes the 30-seed v2 campaign in results_v2; a further argument filters experiments.
        boolean v2 = args.length > 0 && args[0].equals("v2");
        Path out = Paths.get(v2 ? "results_v2" : "results_v3");
        Files.createDirectories(out);
        Set<String> only = new HashSet<>();
        String filt = v2 ? (args.length > 1 ? args[1] : "") : (args.length > 0 ? args[0] : "");
        if (!filt.isBlank()) only.addAll(Arrays.asList(filt.split(",")));
        List<RunSpec> plan = v2 ? plan() : plan3();
        if (!only.isEmpty()) plan.removeIf(r -> !only.contains(r.experiment));
        Path csv = out.resolve("runs.csv");
        Set<String> done = new HashSet<>();
        if (Files.exists(csv)) for (String line : Files.readAllLines(csv)) if (!line.startsWith("run_id")) done.add(line.split(",")[0]);
        List<RunSpec> todo = new ArrayList<>();
        for (RunSpec r : plan) if (!done.contains(r.id())) todo.add(r);
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        System.out.printf("AGWO-VMC CloudSim Plus campaign: %d runs planned, %d already done, %d to run on %d threads%n",
                plan.size(), plan.size() - todo.size(), todo.size(), threads);
        final String header = "run_id,experiment,scenario,method,config,seed,fleet_pms,n_vms,cycles,energy_kwh,slatah,pdm,slav_pct,migrations,failed_migrations,"
                + "mig_energy_kwh,conv_iter95,iters_per_cycle,final_fitness,opt_time_s_median,opt_time_s_max,fe_per_cycle,decision_change_frac,active_hosts_mean,wall_s";
        if (!Files.exists(csv)) Files.writeString(csv, header + System.lineSeparator());
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        AtomicInteger finished = new AtomicInteger(), failed = new AtomicInteger();
        long t0 = System.currentTimeMillis();
        List<Future<?>> fs = new ArrayList<>();
        for (RunSpec r : todo) {
            fs.add(pool.submit(() -> {
                long s0 = System.currentTimeMillis();
                try {
                    Map<String, Double> res = new SimRun(r, out).run();
                    StringBuilder sb = new StringBuilder();
                    sb.append(r.id()).append(',').append(r.experiment).append(',').append(r.scenario).append(',').append(r.method).append(',')
                      .append(r.config.replace(",", ";")).append(',').append(r.seed).append(',').append(r.fleet).append(',').append(3 * r.fleet).append(',').append(r.cycles);
                    for (String k : new String[]{"energy_kwh", "slatah", "pdm", "slav_pct", "migrations", "failed_migrations", "mig_energy_kwh", "conv_iter95",
                            "iters_per_cycle", "final_fitness", "opt_time_s_median", "opt_time_s_max", "fe_per_cycle", "decision_change_frac", "active_hosts_mean"})
                        sb.append(',').append(String.format(Locale.ROOT, "%.8g", res.get(k)));
                    sb.append(',').append(String.format(Locale.ROOT, "%.1f", (System.currentTimeMillis() - s0) / 1000.0));
                    synchronized (Main.class) {
                        Files.writeString(csv, sb + System.lineSeparator(), StandardOpenOption.APPEND);
                    }
                } catch (Throwable e) {
                    failed.incrementAndGet();
                    synchronized (Main.class) {
                        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out.resolve("errors.txt"), StandardCharsets.UTF_8,
                                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
                            pw.println("=== " + r.id()); e.printStackTrace(pw);
                        } catch (IOException ignored) { }
                    }
                }
                int k = finished.incrementAndGet();
                double el = (System.currentTimeMillis() - t0) / 1000.0, eta = el / k * (todo.size() - k);
                String msg = String.format(Locale.ROOT, "%d/%d runs finished (%d failed); elapsed %.1f min; estimated remaining %.1f min; last: %s (%.0f s)",
                        k, todo.size(), failed.get(), el / 60, eta / 60, r.id(), (System.currentTimeMillis() - s0) / 1000.0);
                System.out.println(msg);
                synchronized (Main.class) {
                    try { Files.writeString(out.resolve("progress.txt"), msg + System.lineSeparator()); } catch (IOException ignored) { }
                }
            }));
        }
        for (Future<?> f : fs) f.get();
        pool.shutdown();
        Files.writeString(out.resolve("DONE.txt"), "Campaign finished: " + finished.get() + " runs, " + failed.get() + " failed." + System.lineSeparator());
        System.out.println("Campaign finished. Results are in " + out.toAbsolutePath());
    }

    /**
     * v3 campaign, 30 paired seeds x 4 scenarios: NSGA-II with the knee-point and the AHP-best rule, each without and with
     * the AGWO-VMC local search, and the four recent metaheuristics (LAA, GSO, OOA, DRA) without and with it. The NSGA-II
     * knee-point runs also record the first front and the AGWO-VMC weighted-sum points at cycles 0, 72, 144 and 216.
     * Seeds and workloads are those of the v2 campaign, so the AGWO-VMC and baseline results of results_v2 pair with these.
     */
    static List<RunSpec> plan3() {
        List<RunSpec> L = new ArrayList<>();
        L.add(new RunSpec("smoke3", "dynamic", "NSGAKNEE_LS", "default", 1, 50, 12));
        String[] methods = {"NSGAKNEE_LS", "NSGAAHP_LS", "LAA_LS", "GSO_LS", "OOA_LS", "DRA_LS", "NSGAKNEE", "NSGAAHP", "LAA", "GSO", "OOA", "DRA"};
        for (int s = 1; s <= 30; s++)
            for (String sc : SCEN)
                for (String me : methods) {
                    RunSpec r = new RunSpec("main3", sc, me, "default", s, FLEET, CYCLES);
                    r.logFronts = me.startsWith("NSGAKNEE");
                    r.logIterations = s <= 5;
                    L.add(r);
                }
        return L;
    }

    /** All runs of the 30-seed revision campaign, in priority order (the most important results are produced first). */
    static List<RunSpec> plan() {
        List<RunSpec> L = new ArrayList<>();
        L.add(new RunSpec("smoke", "dynamic", "AGWO", "default", 1, 50, 12));
        // 1) main comparison, 30 paired seeds. "_LS" = the same baseline with the AGWO-VMC local search (like-for-like).
        String[] methods = {"AGWO", "BFD", "GWO_LS", "GA_LS", "PSO_LS", "HHO_LS", "IGWO_LS", "GWO", "GA", "PSO", "HHO", "IGWO", "MGWO", "CGWO"};
        for (int s = 1; s <= 30; s++)
            for (String sc : SCEN)
                for (String me : methods) {
                    RunSpec r = new RunSpec("main", sc, me, "default", s, FLEET, CYCLES);
                    r.logIterations = s <= 5 && (me.equals("AGWO") || me.equals("GWO") || me.equals("GWO_LS") || me.equals("PSO") || me.equals("GA"));
                    L.add(r);
                }
        // 2) ablation, 30 paired seeds: D = feedback-controlled decay, P = softmax encoding, L = local search
        //    (D0P1L1 is identical to GWO_LS of the main comparison and is taken from there)
        for (int s = 1; s <= 30; s++)
            for (String sc : SCEN) {
                for (int D = 1; D >= 0; D--) for (int P = 1; P >= 0; P--) for (int Ls = 1; Ls >= 0; Ls--) {
                    if (D == 1 && P == 1 && Ls == 1) continue;        // full method = AGWO of the main comparison
                    if (D == 0 && P == 1 && Ls == 1) continue;        // = GWO_LS of the main comparison
                    RunSpec r = new RunSpec("ablation", sc, "AGWO", "D" + D + "P" + P + "L" + Ls, s, FLEET, CYCLES);
                    if (D == 0) r.p("decay", "LINEAR").p("gamma", 0.0);
                    if (P == 0) r.p("encoding", "round");
                    if (Ls == 0) r.p("local_search", false);
                    L.add(r);
                }
                L.add(new RunSpec("ablation", sc, "AGWO", "Dg0P1L1", s, FLEET, CYCLES).p("gamma", 0.0));
                L.add(new RunSpec("ablation", sc, "AGWO", "D1P1L1E0", s, FLEET, CYCLES).p("evacuation", false));
                RunSpec w0 = new RunSpec("ablation", sc, "AGWO", "D1P1L1W0", s, FLEET, CYCLES);
                w0.weights = new double[]{0.25, 0.25, 0.25, 0.25};
                L.add(w0);
            }
        // 3) one-at-a-time sensitivity (Dynamic, 10 seeds)
        Map<String, String[]> grid = new LinkedHashMap<>();
        grid.put("lambda", new String[]{"0.5", "0.75", "1.0", "1.25", "1.5", "1.7", "1.9", "2.1", "2.5"});
        grid.put("gamma", new String[]{"0", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"});
        grid.put("k", new String[]{"1", "3", "5", "7", "9"});
        grid.put("beta", new String[]{"0.1", "0.5", "1.0", "1.5", "2", "5", "10", "round"});
        grid.put("t_ls", new String[]{"1", "5", "10", "20"});
        grid.put("l_max", new String[]{"10", "25", "50", "100"});
        for (int s = 1; s <= 10; s++)
            for (Map.Entry<String, String[]> e : grid.entrySet())
                for (String v : e.getValue()) {
                    RunSpec r = new RunSpec("sens", "dynamic", "AGWO", e.getKey() + "-" + v, s, FLEET, CYCLES);
                    if (v.equals("round")) r.p("encoding", "round"); else r.p(e.getKey(), v);
                    r.logIterations = s <= 3 && (e.getKey().equals("k") || e.getKey().equals("beta"));
                    L.add(r);
                }
        // 4) named weight profiles (Dynamic, 10 seeds), AGWO-VMC and GWO with the same local search
        Map<String, double[]> prof = new LinkedHashMap<>();
        prof.put("default", new double[]{0.50, 0.30, 0.15, 0.05});
        prof.put("equal", new double[]{0.25, 0.25, 0.25, 0.25});
        prof.put("sla_first", new double[]{0.30, 0.50, 0.15, 0.05});
        prof.put("migration_averse", new double[]{0.40, 0.25, 0.30, 0.05});
        prof.put("balance_aware", new double[]{0.40, 0.30, 0.10, 0.20});
        prof.put("energy_extreme", new double[]{0.85, 0.10, 0.04, 0.01});
        prof.put("sla_extreme", new double[]{0.10, 0.85, 0.04, 0.01});
        for (int s = 1; s <= 10; s++)
            for (Map.Entry<String, double[]> e : prof.entrySet())
                for (String me : new String[]{"AGWO", "GWO_LS"}) {
                    if (e.getKey().equals("default")) continue;       // default profile = main comparison
                    RunSpec r = new RunSpec("weights_named", "dynamic", me, "w-" + e.getKey(), s, FLEET, CYCLES);
                    r.weights = e.getValue(); L.add(r);
                }
        // 5) scalability (Dynamic, 24 cycles, 5 seeds), 50 to 1,000 hosts
        for (int s = 1; s <= 5; s++)
            for (int size : new int[]{50, 100, 200, 400, 800, 1000})
                for (String me : new String[]{"AGWO", "GWO", "GWO_LS"})
                    L.add(new RunSpec("scal", "dynamic", me, "default", s, size, 24));
        // 6) 50 random weight vectors (uniform on the simplex, fixed seed), Dynamic, 3 seeds
        Random wr = new Random(20261002L);
        for (int v = 1; v <= 50; v++) {
            double[] w = new double[4]; double sum = 0;
            for (int i = 0; i < 4; i++) { w[i] = -Math.log(1 - wr.nextDouble()); sum += w[i]; }
            for (int i = 0; i < 4; i++) w[i] /= sum;
            String cfg = String.format(Locale.ROOT, "rw%02d-%.3f-%.3f-%.3f-%.3f", v, w[0], w[1], w[2], w[3]);
            for (int s = 1; s <= 3; s++)
                for (String me : new String[]{"AGWO", "GWO_LS"}) {
                    RunSpec r = new RunSpec("weights_random", "dynamic", me, cfg, s, FLEET, CYCLES);
                    r.weights = w.clone(); L.add(r);
                }
        }
        // 7) lambda x gamma grid (Dynamic, 5 seeds)
        String[] lg = {"0.5", "0.75", "1.0", "1.25", "1.5", "1.7", "1.9", "2.1", "2.5"}, gg = {"0", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"};
        for (int s = 1; s <= 5; s++)
            for (String l : lg) for (String g : gg)
                L.add(new RunSpec("lamgam", "dynamic", "AGWO", "lambda-" + l + "_gamma-" + g, s, FLEET, CYCLES).p("lambda", l).p("gamma", g));
        return L;
    }
}
