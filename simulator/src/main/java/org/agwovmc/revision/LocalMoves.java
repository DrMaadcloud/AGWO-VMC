package org.agwovmc.revision;

import java.util.Random;

/** Problem-specific neighbourhood used by the local search (e.g. host evacuation). Returns null if no move applies. */
public interface LocalMoves {
    int[] propose(int[] placement, Random rng);
}
