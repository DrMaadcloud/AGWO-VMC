package org.agwovmc.revision;

/** Receives per-iteration diagnostics (only called on logged cycles, see RunLogger). */
public interface IterationListener {
    void onIteration(int iter, double fBest, double a, double g, double gbar,
                     double divContinuous, double divPlacement, double repairRate);
}
