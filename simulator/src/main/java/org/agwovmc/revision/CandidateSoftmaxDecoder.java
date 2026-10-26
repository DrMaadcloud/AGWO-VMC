package org.agwovmc.revision;

import java.util.Random;

/**
 * Probabilistic decoding (Section 4.3, Eq. 13). Positions X_j are continuous PM coordinates in [0, m-1]
 * (0-based version of [1, m]). Candidate set C_j = the c-1 PMs nearest to X_j (a contiguous index window)
 * plus one PM drawn uniformly from the remaining PMs. Score s_jp = -|X_j - p|; P(p) = softmax(s/beta) over C_j.
 * beta <= 0 means hard rounding to the nearest index.
 * Because the weights inside the window are geometric on each side of X_j, sampling is done in O(1)
 * by inverse transform, which gives exactly the same distribution as enumerating the candidates.
 */
public final class CandidateSoftmaxDecoder {
    private final int m, c;
    private final double beta, q, logq;

    public CandidateSoftmaxDecoder(int numPms, int candidateSize, double beta) {
        this.m = numPms; this.c = Math.max(2, Math.min(candidateSize, numPms)); this.beta = beta;
        this.q = beta > 0 ? Math.exp(-1.0 / beta) : 0; this.logq = beta > 0 ? Math.log(q) : 0;
    }

    public int[] decode(double[] x, Random rng) {
        int n = x.length;
        int[] out = new int[n];
        int win = Math.min(c - 1, m);
        for (int j = 0; j < n; j++) {
            double xj = Math.min(m - 1, Math.max(0, x[j]));
            if (beta <= 0) { out[j] = (int) Math.round(xj); continue; }
            int a = (int) Math.floor(xj - (win - 1) / 2.0 + 0.5);
            a = Math.max(0, Math.min(a, m - win));
            int b = a + win - 1;
            int fl = (int) Math.floor(xj), ce = fl + 1;
            int nL = fl >= a ? Math.min(fl, b) - a + 1 : 0;
            int nR = ce <= b ? b - Math.max(ce, a) + 1 : 0;
            double eL = nL > 0 ? Math.exp(-(xj - Math.min(fl, b)) / beta) : 0;
            double eR = nR > 0 ? Math.exp(-(Math.max(ce, a) - xj) / beta) : 0;
            double SL = nL > 0 ? eL * geo(nL) : 0, SR = nR > 0 ? eR * geo(nR) : 0;
            int r = -1; double wr = 0;
            if (m > win) {
                r = rng.nextInt(m - win); if (r >= a) r += win;      // uniform outside [a, b]
                wr = Math.exp(-Math.abs(xj - r) / beta);
            }
            double u = rng.nextDouble() * (SL + SR + wr);
            int pick;
            if (u < SL) pick = Math.min(fl, b) - offset(u / eL, nL);
            else if (u < SL + SR) pick = Math.max(ce, a) + offset((u - SL) / eR, nR);
            else pick = r;
            out[j] = pick;
        }
        return out;
    }

    /** sum_{k=0}^{n-1} q^k */
    private double geo(int n) { return q >= 1 ? n : (1 - Math.pow(q, n)) / (1 - q); }

    /** Smallest k with sum_{i=0}^{k} q^i > v, clamped to [0, n-1]. */
    private int offset(double v, int n) {
        if (q <= 0) return 0;
        double arg = 1 - v * (1 - q);
        int k = arg <= 0 ? n - 1 : (int) Math.floor(Math.log(arg) / logq);
        return Math.max(0, Math.min(n - 1, k));
    }
}
