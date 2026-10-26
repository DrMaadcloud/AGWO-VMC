package org.agwovmc.sim;

import org.agwovmc.revision.*;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/** Builds the CloudSim Plus data centre for one run, performs the consolidation every 300 s and measures the metrics. */
public final class SimRun {
    static boolean DEBUG = Boolean.getBoolean("agwo.debug"); int dbg = 0;
    static final int HOST_SLOTS = 32;
    static final double HOST_BW = 1000, MIG_BW_SHARE = 0.5, P_NET_W = 10.0, DELTA = 1.0;
    private final RunSpec spec; private final Path outDir;
    public SimRun(RunSpec spec, Path outDir) { this.spec = spec; this.outDir = outDir; }

    public Map<String, Double> run() throws IOException {
        final int m = spec.fleet, n = 3 * m, C = spec.cycles;
        Workload wl = new Workload(spec.scenario, n, C, SeedPlan.workloadSeed(spec.seed, spec.scenario));
        Random algo = new Random(SeedPlan.algorithmSeed(spec.seed, spec.scenario, spec.method + spec.config));

        org.cloudsimplus.util.Log.setLevel(ch.qos.logback.classic.Level.OFF);
        CloudSimPlus sim = new CloudSimPlus(1.0);
        List<Host> hosts = new ArrayList<>();
        double[] cap = new double[m]; int[] htype = new int[m];
        for (int h = 0; h < m; h++) {
            htype[h] = h % 2;   // half G4, half G5
            double mips = Power.MIPS_PER_PE[htype[h]];
            // CloudSim Plus 8.5 reserves one host PE per VM PE at nominal MIPS (no CPU over-subscription). Each host is therefore
            // given HOST_SLOTS VM slots of one core's MIPS; the physical two-core capacity (cap[h]) is enforced by the
            // consolidation layer (Problem/Repair) and by the power and SLA measurement below.
            List<Pe> pes = new ArrayList<>();
            for (int p = 0; p < HOST_SLOTS; p++) pes.add(new PeSimple(mips));
            hosts.add(new HostSimple(8192, (long) HOST_BW, 1_000_000, pes));
            cap[h] = 2 * mips;
        }
        // initial placement: three VMs per host in index order (the usual starting point of the consolidation study)
        VmAllocationPolicySimple alloc = new VmAllocationPolicySimple();
        alloc.setFindHostForVmFunction((pol, vm) -> Optional.of(hosts.get((int) (vm.getId() / 3) % m)));
        Datacenter dc = new DatacenterSimple(sim, hosts, alloc);
        dc.setSchedulingInterval(Workload.CYCLE_S);
        DatacenterBroker broker = new DatacenterBrokerSimple(sim);
        List<Vm> vms = new ArrayList<>(); List<Cloudlet> cls = new ArrayList<>();
        double[] vmMips = new double[n], vmRam = new double[n];
        for (int j = 0; j < n; j++) {
            int t = wl.type[j];
            vmMips[j] = Workload.VM_MIPS[t]; vmRam[j] = Workload.VM_RAM[t];
            Vm vm = new VmSimple(vmMips[j], 1).setRam(Workload.VM_RAM[t]).setBw(100).setSize(1000);
            Cloudlet cl = new CloudletSimple(-1, 1, new ScenarioUtilization(wl, j));
            cl.setUtilizationModelRam(new UtilizationModelFull());
            cl.setUtilizationModelBw(new UtilizationModelFull());
            cl.setVm(vm);
            vms.add(vm); cls.add(cl);
        }
        broker.submitVmList(vms); broker.submitCloudletList(cls);
        sim.startSync();
        for (int k = 0; k < 20 && vms.stream().anyMatch(v -> !v.isCreated()); k++) advance(sim, 1.0);
        Map<Host, Integer> hIdx = new IdentityHashMap<>();
        for (int h = 0; h < m; h++) hIdx.put(hosts.get(h), h);
        double activeSum = 0;

        double energyWs = 0, migEnergyJ = 0, convSum = 0, iterSum = 0, fitSum = 0, changeSum = 0;
        long migrations = 0, failedMig = 0, fePerCycle = 0;
        double[] activeTime = new double[m], overTime = new double[m];
        double[] reqMipsS = new double[n], degMipsS = new double[n];
        double[] optTimes = new double[C];
        int[] prev = null;
        PrintWriter iterLog = null;
        if (spec.logIterations) {
            Files.createDirectories(outDir.resolve("iters"));
            iterLog = new PrintWriter(Files.newBufferedWriter(outDir.resolve("iters").resolve(spec.id() + ".csv"), StandardCharsets.UTF_8));
            iterLog.println("cycle,iter,best_fitness,a,g,gbar,div_cont,div_place,repair_rate");
        }
        for (int c = 0; c < C; c++) {
            double cycleStart = c * (double) Workload.CYCLE_S;
            if (sim.clock() < cycleStart) advance(sim, cycleStart - sim.clock());
            int[] cur = new int[n];
            double[] dem = new double[n];
            for (int j = 0; j < n; j++) {
                Host hh = vms.get(j).getHost();
                cur[j] = hIdx.getOrDefault(hh, -1);
                dem[j] = wl.demand[j][c] * vmMips[j];
            }
            fixUnplaced(cur, cap, dem, vmRam);
            if (DEBUG) System.err.println("cycle " + c + " clock=" + sim.clock() + " vm0 req=" + vms.get(0).getCurrentRequestedMips() + " dem0=" + dem[0] + " host0 avail=" + hosts.get(0).getVmScheduler().getTotalAvailableMips() + " cl0 status=" + cls.get(0).getStatus() + " created=" + broker.getVmCreatedList().size() + " failed=" + broker.getVmFailedList().size() + " waiting=" + broker.getVmWaitingList().size() + " exec=" + broker.getCloudletSubmittedList().size());
            Problem pr = new Problem(cap, htype, dem, vmRam, cur, spec.weights);
            final int cyc = c;
            final PrintWriter il = iterLog;
            IterationListener lis = (il != null && c % 72 == 0) ? (it, fb, a, g, gb, dc1, dp, rr) ->
                    il.println(String.format(Locale.ROOT, "%d,%d,%.8f,%.5f,%.6g,%.5f,%.5f,%.5f,%.5f", cyc, it, fb, a, g, gb, dc1, dp, rr)) : null;
            long t0 = System.nanoTime();
            int[] target; OptimizerResult res = null;
            final boolean plusLs = spec.method.endsWith("_LS");
            final String base = plusLs ? spec.method.substring(0, spec.method.length() - 3) : spec.method;
            switch (base) {
                case "BFD": target = Pabfd.place(pr); break;
                case "GA": res = new GaOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "PSO": res = new PsoOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "HHO": res = new HhoOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "LAA": res = new LaaOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "GSO": res = new GsoOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "OOA": res = new OoaOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "DRA": res = new DraOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                case "NSGAKNEE": case "NSGAAHP": {
                    NsgaOptimizer ns = new NsgaOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5), base.equals("NSGAKNEE") ? NsgaOptimizer.Pick.KNEE : NsgaOptimizer.Pick.AHP)
                            .withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50));
                    res = ns.optimise(pr, pr, cur, algo, lis); target = res.placement;
                    if (spec.logFronts && c % 72 == 0) dumpFront(c, pr, cur, ns.frontObjectives());
                    break;
                }
                case "IGWO": res = new IgwoOptimizer(50, 12000, ci("c", 32), cd("beta", 1.5)).withLocalSearch(plusLs ? pr : null, ci("t_ls", 5), ci("l_max", 50)).optimise(pr, pr, cur, algo, lis); target = res.placement; break;
                default: {
                    GwoFamilyOptimizer.Config gc = gwoConfig();
                    if (gc.localSearch && Boolean.parseBoolean(spec.param("evacuation", "true"))) gc.evacuation = pr;
                    res = new GwoFamilyOptimizer(gc).optimise(pr, pr, cur, algo, lis); target = res.placement;
                }
            }
            optTimes[c] = (System.nanoTime() - t0) / 1e9;
            if (res != null) { convSum += res.convIter95; iterSum += res.iterations; fePerCycle = res.evaluations; }
            fitSum += pr.evaluate(target);

            // apply the plan through CloudSim Plus live migrations, in up to four rounds
            Map<Vm, Host> pending = new LinkedHashMap<>();
            for (int j = 0; j < n; j++) if (target[j] != cur[j]) pending.put(vms.get(j), hosts.get(target[j]));
            for (int round = 0; round < 4 && !pending.isEmpty(); round++) {
                for (Host hh : hosts) resyncPes(hh);
                Iterator<Map.Entry<Vm, Host>> itp = pending.entrySet().iterator();
                boolean any = false;
                while (itp.hasNext()) {
                    Map.Entry<Vm, Host> e = itp.next();
                    Vm vm = e.getKey(); Host th = e.getValue();
                    if (vm.isInMigration() || vm.getHost() == th) { itp.remove(); continue; }
                    if (th.isSuitableForVm(vm)) {
                        dc.requestVmMigration(vm, th); itp.remove(); any = true; migrations++;
                        int j = vms.indexOf(vm);
                        double tMig = vmRam[j] * 8.0 / (HOST_BW * MIG_BW_SHARE);
                        degMipsS[j] += 0.1 * dem[j] * tMig;
                        migEnergyJ += tMig * P_NET_W * DELTA;
                    } else if (DEBUG && dbg++ < 2 + 2 * c) System.err.println("c" + c + " r" + round + " " + th.getSuitabilityFor(vm) + " vmReqMips=" + vm.getCurrentRequestedMips() + " hostFreePes=" + th.getFreePesNumber() + " busyPes=" + th.getBusyPesNumber() + " vmsOnHost=" + th.getVmList().size() + " migIn=" + th.getVmsMigratingIn().size() + " active=" + th.isActive() + " suit=" + th.getSuitabilityFor(vm).forPes() + "/" + th.getSuitabilityFor(vm).forRam() + " hostAvailMips=" + th.getVmScheduler().getTotalAvailableMips() + " ram=" + th.getRam().getAvailableResource());
                }
                advance(sim, 30.0);
                if (!any) break;
            }
            failedMig += pending.size();

            // measurement for this cycle, on the placement CloudSim actually holds
            double[] load = new double[m]; int[] cnt = new int[m]; int[] now = new int[n];
            for (int j = 0; j < n; j++) {
                Host hh = vms.get(j).getHost();
                int h = hIdx.getOrDefault(hh, cur[j]);
                now[j] = h; load[h] += dem[j]; cnt[h]++;
                reqMipsS[j] += dem[j] * Workload.CYCLE_S;
            }
            for (int h = 0; h < m; h++) {
                if (cnt[h] == 0) continue;
                activeSum++;
                double u = load[h] / cap[h];
                energyWs += Power.watts(htype[h], u) * Workload.CYCLE_S;
                activeTime[h] += Workload.CYCLE_S;
                if (u > 1.0) overTime[h] += Workload.CYCLE_S;
            }
            if (prev != null) { int d = 0; for (int j = 0; j < n; j++) if (now[j] != prev[j]) d++; changeSum += (double) d / n; }
            prev = now;
        }
        if (iterLog != null) iterLog.close();
        sim.terminate();

        double slatah = 0; int act = 0;
        for (int h = 0; h < m; h++) if (activeTime[h] > 0) { slatah += overTime[h] / activeTime[h]; act++; }
        slatah /= Math.max(1, act);
        double pdm = 0; for (int j = 0; j < n; j++) pdm += reqMipsS[j] > 0 ? degMipsS[j] / reqMipsS[j] : 0; pdm /= n;
        double[] ot = optTimes.clone(); Arrays.sort(ot);
        Map<String, Double> r = new LinkedHashMap<>();
        r.put("energy_kwh", energyWs / 3.6e6);
        r.put("slatah", slatah); r.put("pdm", pdm); r.put("slav_pct", 100.0 * slatah * pdm);
        r.put("migrations", (double) migrations); r.put("failed_migrations", (double) failedMig);
        r.put("mig_energy_kwh", migEnergyJ / 3.6e6);
        r.put("conv_iter95", convSum / C); r.put("iters_per_cycle", iterSum / C);
        r.put("final_fitness", fitSum / C);
        r.put("opt_time_s_median", ot[C / 2]); r.put("opt_time_s_max", ot[C - 1]);
        r.put("fe_per_cycle", (double) fePerCycle);
        r.put("decision_change_frac", changeSum / Math.max(1, C - 1));
        r.put("active_hosts_mean", activeSum / C);
        return r;
    }

    /** Weight profiles of Section 7.3 used to trace the weighted-sum (AGWO-VMC) points of the front. */
    static final String[] PROF = {"default", "equal", "sla_first", "migration_averse", "balance_aware", "energy_extreme", "sla_extreme"};
    static final double[][] PROFW = {{0.50, 0.30, 0.15, 0.05}, {0.25, 0.25, 0.25, 0.25}, {0.30, 0.50, 0.15, 0.05}, {0.40, 0.25, 0.30, 0.05},
            {0.40, 0.30, 0.10, 0.20}, {0.85, 0.10, 0.04, 0.01}, {0.10, 0.85, 0.04, 0.01}};

    /**
     * On the same demand snapshot and incumbent, writes the NSGA-II first front and the placements AGWO-VMC returns under the
     * seven weight profiles (objective vectors in the normalised space of Eq. 8), for the hypervolume and IGD of Table 7.
     * AGWO-VMC uses its own random stream, so the NSGA-II run itself is not affected.
     */
    private void dumpFront(int c, Problem pr, int[] cur, List<double[]> front) throws IOException {
        Path dir = outDir.resolve("fronts");
        Files.createDirectories(dir);
        StringBuilder sb = new StringBuilder();
        for (double[] o : front) sb.append(String.format(Locale.ROOT, "%d,NSGA,%.8g,%.8g,%.8g,%.8g%n", c, o[0], o[1], o[2], o[3]));
        for (int k = 0; k < PROF.length; k++) {
            Problem pw = pr.withWeights(PROFW[k]);
            GwoFamilyOptimizer.Config gc = new GwoFamilyOptimizer.Config();
            gc.evacuation = pw;
            Random r = new Random(SeedPlan.algorithmSeed(spec.seed, spec.scenario, "AGWOFRONT" + c + "-" + k));
            long t0 = System.nanoTime();
            OptimizerResult a = new GwoFamilyOptimizer(gc).optimise(pw, pw, cur, r, null);
            double sec = (System.nanoTime() - t0) / 1e9;
            double[] o = pr.objectives(a.placement);
            sb.append(String.format(Locale.ROOT, "%d,AGWO-%s,%.8g,%.8g,%.8g,%.8g,%.3f%n", c, PROF[k], o[0], o[1], o[2], o[3], sec));
        }
        Path file = dir.resolve(spec.id() + ".csv");
        if (!Files.exists(file)) Files.writeString(file, "cycle,source,e,sla,mig,cv,opt_s" + System.lineSeparator());
        Files.writeString(file, sb.toString(), java.nio.file.StandardOpenOption.APPEND);
    }

    /** CloudSim Plus 8.5.7 leaks busy-PE counts after live migrations; the PE status is rebuilt from the VMs the host holds. */
    static void resyncPes(Host h) {
        long need = 0;
        for (Vm v : h.getVmList()) need += v.getPesNumber();
        for (Vm v : h.getVmsMigratingIn()) if (!h.getVmList().contains(v)) need += v.getPesNumber();
        List<Pe> pl = h.getPeList();
        org.cloudsimplus.hosts.HostAbstract ha = (org.cloudsimplus.hosts.HostAbstract) h;
        ha.setPeStatus(pl, Pe.Status.FREE);
        ha.setPeStatus(pl.subList(0, (int) Math.min(need, pl.size())), Pe.Status.BUSY);
    }
    /** runFor() in CloudSim Plus 8.5 stops after the first batch of events, so the clock is advanced in a loop. */
    static void advance(CloudSimPlus sim, double dt) {
        double until = sim.clock() + dt; int guard = 0;
        while (sim.clock() < until - 1e-9 && guard++ < 1_000_000) sim.runFor(until - sim.clock());
    }
    private void fixUnplaced(int[] cur, double[] cap, double[] dem, double[] ram) {
        for (int j = 0; j < cur.length; j++) if (cur[j] < 0) cur[j] = (j / 3) % cap.length;
    }
    private int ci(String k, int d) { return Integer.parseInt(spec.param(k, String.valueOf(d))); }
    private double cd(String k, double d) { return Double.parseDouble(spec.param(k, String.valueOf(d))); }

    private GwoFamilyOptimizer.Config gwoConfig() {
        GwoFamilyOptimizer.Config c = new GwoFamilyOptimizer.Config();
        c.maxEvaluations = 12000; c.populationSize = 50;
        String base = spec.method.endsWith("_LS") ? spec.method.substring(0, spec.method.length() - 3) : spec.method;
        switch (base) {
            case "GWO": c.decay = ImprovementFeedback.Decay.LINEAR; c.gamma = 0; c.localSearch = false; break;
            case "MGWO": c.decay = ImprovementFeedback.Decay.MGWO; c.gamma = 0; c.localSearch = false; break;
            case "CGWO": c.decay = ImprovementFeedback.Decay.LINEAR; c.gamma = 0; c.localSearch = false; c.chaotic = true; break;
            default: break;   // AGWO
        }
        c.decay = ImprovementFeedback.Decay.valueOf(spec.param("decay", c.decay.name()));
        c.lambda = cd("lambda", c.lambda); c.gamma = cd("gamma", c.gamma); c.k = ci("k", c.k);
        c.beta = "round".equals(spec.param("encoding", "softmax")) ? 0.0 : cd("beta", c.beta);
        c.candidateSize = ci("c", c.candidateSize); c.tLs = ci("t_ls", c.tLs); c.lMax = ci("l_max", c.lMax);
        if (spec.method.endsWith("_LS")) c.localSearch = true;   // like-for-like baseline: same local search as AGWO-VMC
        c.localSearch = Boolean.parseBoolean(spec.param("local_search", String.valueOf(c.localSearch)));
        return c;
    }
}
