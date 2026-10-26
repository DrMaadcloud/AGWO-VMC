package org.agwovmc.sim;

import org.cloudsimplus.utilizationmodels.UtilizationModelAbstract;

/** CPU utilisation model of one cloudlet: follows the scenario workload of its VM. */
public final class ScenarioUtilization extends UtilizationModelAbstract {
    private final Workload w; private final int j;
    public ScenarioUtilization(Workload w, int j) { super(); this.w = w; this.j = j; }
    @Override protected double getUtilizationInternal(double time) { return w.at(j, time); }
}
