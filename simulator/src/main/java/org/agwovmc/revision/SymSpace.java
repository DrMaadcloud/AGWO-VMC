package org.agwovmc.revision;

import java.util.Random;

/**
 * Search space shared by the four recent metaheuristics (LAA, GSO, OOA, DRA). Their operators were defined and tested on
 * box domains centred on the origin, and several of them (division and multiplication by the best solution, scaling of
 * positions, sinusoidal terms) are meaningful only there. Each algorithm therefore searches z in [-1, 1]^n, which is
 * mapped linearly to the PM coordinates x = (z + 1)(m - 1)/2 in [0, m - 1] before the softmax decoding and repair that
 * every method uses. Initialisation (warm start from the incumbent), budget accounting and the elitist archive are those
 * of the other population methods.
 */
final class SymSpace {
    static final double LB = -1, UB = 1;
    final int n, m;
    final Fitness f; final Repair repair; final CandidateSoftmaxDecoder dec; final EvalBudget budget; final ElitistArchive arch;
    final Random rng;

    SymSpace(Fitness f, Repair repair, int c, double beta, long maxEvals, Random rng) {
        this.f = f; this.repair = repair; this.rng = rng;
        n = f.numVms(); m = f.numPms();
        dec = new CandidateSoftmaxDecoder(m, c, beta); budget = new EvalBudget(maxEvals); arch = new ElitistArchive();
    }

    double[] initial(int[] incumbent, int agent) {
        double[] z = new double[n];
        for (int j = 0; j < n; j++) z[j] = toZ(Init.coord(incumbent, agent, j, m, rng));
        return z;
    }
    double toZ(double x) { return 2 * x / (m - 1) - 1; }
    double[] toZ(int[] p) { double[] z = new double[p.length]; for (int j = 0; j < p.length; j++) z[j] = toZ(p[j]); return z; }
    void clamp(double[] z) { for (int j = 0; j < z.length; j++) z[j] = Math.min(UB, Math.max(LB, Double.isFinite(z[j]) ? z[j] : 0)); }

    /** Decodes, repairs and evaluates z (one budget unit). Returns +inf when the budget is exhausted. */
    double eval(double[] z) {
        if (budget.exhausted()) return Double.POSITIVE_INFINITY;
        clamp(z);
        double[] x = new double[n];
        for (int j = 0; j < n; j++) x[j] = (z[j] + 1) * (m - 1) / 2.0;
        int[] p = dec.decode(x, rng); repair.repair(p); budget.take();
        double v = f.evaluate(p); arch.offer(p, v); return v;
    }

    void offerIncumbent(int[] incumbent) {
        if (incumbent != null && budget.take()) { int[] inc = incumbent.clone(); repair.repair(inc); arch.offer(inc, f.evaluate(inc)); }
    }

    /** Local search on the archive; afterwards the archive best replaces the worst population member. */
    void localSearch(LocalMoves moves, int lMax, double[][] Z, double[] F) {
        LocalSearch.apply(arch, f, repair, lMax, moves, budget, rng);
        int w = 0; for (int i = 1; i < F.length; i++) if (F[i] > F[w]) w = i;
        if (arch.fitness() < F[w]) { Z[w] = toZ(arch.best()); F[w] = arch.fitness(); }
    }

    /** Mantegna's algorithm for a Levy-stable step with index b. */
    private double levyB = Double.NaN, levySigma;
    double levy(double b) {
        if (b != levyB) { levyB = b; levySigma = Math.pow(gamma(1 + b) * Math.sin(Math.PI * b / 2) / (gamma((1 + b) / 2) * b * Math.pow(2, (b - 1) / 2)), 1 / b); }
        double sigma = levySigma;
        return rng.nextGaussian() * sigma / Math.pow(Math.abs(rng.nextGaussian()) + 1e-300, 1 / b);
    }
    static double gamma(double x) {
        double[] g = {676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
                12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
        if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gamma(1 - x));
        x -= 1; double a = 0.99999999999980993; double t = x + 7.5;
        for (int i = 0; i < 8; i++) a += g[i] / (x + i + 1);
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
    }
}
