package org.agwovmc.revision;

import java.util.ArrayList;
import java.util.List;

/**
 * Trajectory-based improvement feedback (Section 4.2, Eqs. 11-12).
 * H[s] = F_best after iteration s (H[0] = best initial placement).
 * g(t) = (H[t-1-k_t] - H[t-1]) / k_t, k_t = min(k, t-1), g(t) = 0 if k_t = 0.
 * gbar(t) = g(t) / G(t) with G(t) the running maximum, gbar = 0 if G = 0.
 * a(t) = 2 (1 - t/tmax)^lambda (1 - gamma * gbar(t)).
 */
public final class ImprovementFeedback {
    public enum Decay { LINEAR, POWER, MGWO }

    private final int k;
    private final double gamma, lambda;
    private final int tMax;
    private final Decay decay;
    private final List<Double> history = new ArrayList<>();
    private double runningMax = 0.0;
    private double lastG, lastGbar;

    public ImprovementFeedback(Decay decay, double lambda, double gamma, int k, int tMax) {
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.decay = decay; this.lambda = lambda; this.gamma = gamma; this.k = k; this.tMax = tMax;
    }

    /** Record F_best after an iteration (call once with the initial value, then once per iteration). */
    public void record(double fBest) {
        if (!history.isEmpty() && fBest > history.get(history.size() - 1) + 1e-12)
            throw new IllegalStateException("F_best increased: the archive is not elitist (violates Lemma 1)");
        history.add(fBest);
    }

    /** Control coefficient for iteration t (1-based). Uses history up to t-1. */
    public double coefficient(int t) {
        int kt = Math.min(k, t - 1);
        double g = 0.0;
        if (kt > 0) {
            int n = history.size();          // n == t when called correctly
            double older = history.get(n - 1 - kt), newer = history.get(n - 1);
            g = (older - newer) / kt;
        }
        if (g < 0) g = 0;                    // cannot happen with an elitist archive
        runningMax = Math.max(runningMax, g);
        double gbar = runningMax > 0 ? g / runningMax : 0.0;
        lastG = g; lastGbar = gbar;
        double x = Math.min(1.0, Math.max(0.0, (double) t / tMax));
        switch (decay) {
            case LINEAR: return 2.0 * (1.0 - x);                       // standard GWO, Eq. 10
            case MGWO:   return 2.0 * (1.0 - x * x);                    // non-linear decay of mGWO [30]
            default:     return 2.0 * Math.pow(1.0 - x, lambda) * (1.0 - gamma * gbar);
        }
    }

    public double lastG() { return lastG; }
    public double lastGbar() { return lastGbar; }
}
