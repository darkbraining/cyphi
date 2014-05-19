#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute
~~~~~~~

Methods for computing concepts, constellations, and integrated information of
subsystems.
"""

import functools
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from . import utils, options, validate
from .subsystem import Subsystem
from .models import Cut, Mechanism, Mice, Concept, BigMip
from .network import Network
from .constants import DIRECTIONS, PAST, FUTURE, MAXMEM, memory
from .lru_cache import lru_cache


_mice_cache = {}


def _get_mice_key(subsystem, direction, mechanism):
    # We use the hash of the subsystem, the direction, and the mechanism
    # indices as the cache key
    return (hash(subsystem), direction, tuple(n.index for n in mechanism))


def _cache_mice(subsystem, direction, mechanism, mice):
    key = _get_mice_key(subsystem, direction, mechanism)
    if key not in _mice_cache:
        _mice_cache[key] = mice


def _get_cached_mice(subsystem, direction, mechanism):
    """Return a cached MICE if there is one and the cut doesn't affect it.

    Return False otherwise."""
    # Hash of subsystem doesn't include cut
    key = _get_mice_key(subsystem, direction, mechanism)
    if key in _mice_cache:
        cached = _mice_cache[key]
        # If we've already calculated the core cause for this mechanism
        # with no cut, then we don't need to recalculate it with the cut if
        #   - all mechanism nodes are severed, or
        #   - all the cached cause's purview nodes are intact.
        if (direction == DIRECTIONS[PAST] and
            (all([nodes in subsystem.cut.severed for nodes in mechanism]) or
             all([nodes in subsystem.cut.intact for nodes in
                  cached.purview]))):
            return cached
        # If we've already calculated the core cause for this mechanism
        # with no cut, then we don't need to recalculate it with the cut if
        #   - all mechanism nodes are intact, or
        #   - all the cached effect's purview nodes are severed.
        if (direction == DIRECTIONS[FUTURE] and
            (all([nodes in subsystem.cut.intact for nodes in mechanism]) or
             all([nodes in subsystem.cut.severed for nodes in
                  cached.purview]))):
            return cached
    return False


# TODO update docs
def find_mice(subsystem, direction, mechanism):
    """Return the maximally irreducible cause or effect for a mechanism.

    Args:
        direction (str): The temporal direction, specifying cause or
            effect.
        mechanism (tuple(Node)): The mechanism to be tested for
            irreducibility.

    Returns:
        :class:`cyphi.models.Mice`

    .. note::
        Strictly speaking, the MICE is a pair of repertoires: the core
        cause repertoire and core effect repertoire of a mechanism, which
        are maximally different than the unconstrained cause/effect
        repertoires (*i.e.*, those that maximize |phi|). Here, we return
        only information corresponding to one direction, |past| or
        |future|, i.e., we return a core cause or core effect, not the pair
        of them.
    """
    # Return a cached MICE if we were given a cache and there's a hit
    cached_mice = _get_cached_mice(subsystem, direction, mechanism)
    if cached_mice:
        return cached_mice

    validate.direction(direction)
    # Set defaults for a reducible MICE
    mip_max = None
    phi_max = float('-inf')
    maximal_purview = None
    maximal_repertoire = None

    purviews = utils.powerset(subsystem.nodes)

    def not_trivially_reducible(purview):
        if direction == DIRECTIONS[PAST]:
            return subsystem._all_connect_to_any(purview, mechanism)
        elif direction == DIRECTIONS[FUTURE]:
            return subsystem._all_connect_to_any(mechanism, purview)

    # Filter out trivially reducible purviews if a connectivity matrix was
    # provided
    purviews = filter(not_trivially_reducible, purviews)

    # Loop over filtered purviews in this candidate set and find the
    # purview over which phi is maximal.
    for purview in purviews:
        mip = subsystem.find_mip(direction, mechanism, purview)
        if mip:
            # Take the purview with higher phi, or if phi is equal, take
            # the larger one (exclusion principle).
            if mip.phi > phi_max or (utils.phi_eq(mip.phi, phi_max) and
                                     len(purview) > len(maximal_purview)):
                mip_max = mip
                phi_max = mip.phi
                maximal_purview = purview
                maximal_repertoire = mip.unpartitioned_repertoire

    # If there was no MIP, then phi is zero.
    if phi_max == float('-inf'):
        phi_max = 0
    # Build the Mice.
    mice = Mice(direction=direction,
                mechanism=mechanism,
                purview=maximal_purview,
                repertoire=maximal_repertoire,
                mip=mip_max,
                phi=phi_max)
    # Cache it if it's not already in there.
    _cache_mice(subsystem, direction, mechanism, mice)
    return mice


def core_cause(subsystem, mechanism):
    """Returns the core cause repertoire of a mechanism.

    Alias for :func:`find_mice` with ``direction`` set to |past|."""
    return find_mice(subsystem, 'past', mechanism)


# TODO! don't use these internally
def core_effect(subsystem, mechanism):
    """Returns the core effect repertoire of a mechanism.

    Alias for :func:`find_mice` with ``direction`` set to |past|."""
    return find_mice(subsystem, 'future', mechanism)


def phi_max(subsystem, mechanism):
    """Return the |phi_max| of a mechanism.

    This is the maximum of |phi| taken over all possible purviews."""
    return min(core_cause(subsystem, mechanism).phi,
               core_effect(subsystem, mechanism).phi)


# XXX: re-cache this after implementing builtin-cuts
def _concept(subsystem, mechanism, hash):
    """Returns the concept specified by a mechanism.

    The output is "persistently cached" (saved to the disk for later access to
    avoid recomputation).
    Cache the output using the normal form of the multiset of the mechanism
    nodes' Markov blankets (not the mechanism itself). This results in more
    cache hits, since the output depends only on the causual properties of the
    nodes. See the marbl documentation.
    """
    # If any node in the mechanism either has no inputs from the subsystem
    # or has no outputs to the subsystem, then the mechanism is necessarily
    # reducible and cannot be a concept (since removing that node would
    # make no difference to at least one of the MICEs).
    if not (subsystem._all_connect_to_any(mechanism, subsystem.nodes) and
            subsystem._any_connect_to_all(subsystem.nodes, mechanism)):
        return None

    past_mice = core_cause(subsystem, mechanism)
    future_mice = core_effect(subsystem, mechanism)
    phi = min(past_mice.phi, future_mice.phi)

    if phi < options.EPSILON:
        return None
    return Concept(
        mechanism=mechanism,
        location=np.array([
            subsystem.expand_cause_repertoire(past_mice.mechanism,
                                              past_mice.purview,
                                              past_mice.repertoire),
            subsystem.expand_effect_repertoire(future_mice.mechanism,
                                               future_mice.purview,
                                               future_mice.repertoire)]),
        phi=phi,
        cause=past_mice,
        effect=future_mice)


@functools.wraps(_concept)
def concept(subsystem, mechanism):
    return _concept(subsystem, mechanism, hash(mechanism))


def constellation(subsystem):
    """Return the conceptual structure of this subsystem."""
    concepts = [concept(subsystem, Mechanism(subset)) for subset in
                utils.powerset(subsystem.nodes)]
    # Filter out non-concepts
    return tuple(filter(None, concepts))


@lru_cache(maxmem=MAXMEM)
def concept_distance(c1, c2):
    """Return the distance between two concepts in concept-space.

    Args:
        c1 (Mice): the first concept
        c2 (Mice): the second concept

    Returns:
        The distance between the two concepts in concept-space.
    """
    return sum([utils.hamming_emd(c1.location[PAST],
                                  c2.location[PAST]),
                utils.hamming_emd(c1.location[FUTURE],
                                  c2.location[FUTURE])])


def _constellation_distance_simple(C1, C2, null_concept):
    """Return the distance between two constellations in concept-space,
    assuming the only difference between them is that some concepts have
    disappeared."""
    # Make C1 refer to the bigger constellation
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c for c in C1 if c not in C2]
    return sum(c.phi * concept_distance(c, null_concept) for c in destroyed)


def _constellation_distance_emd(C1, C2, unique_C1, unique_C2, null_concept):
    """Return the distance between two constellations in concept-space,
    using the generalized EMD."""
    shared_concepts = [c for c in C1 if c in C2]
    # Construct null concept and list of all unique concepts.
    all_concepts = shared_concepts + unique_C1 + unique_C2 + [null_concept]
    # Construct the two phi distributions.
    d1, d2 = [[c.phi if c in constellation else 0 for c in all_concepts]
              for constellation in (C1, C2)]
    # Calculate how much phi disappeared and assign it to the null concept
    # (the null concept is the last element in the distribution).
    residual = sum(d1) - sum(d2)
    if residual > 0:
        d2[-1] = residual
    if residual < 0:
        d1[-1] = residual
    # Generate the ground distance matrix.
    distance_matrix = np.array([
        [concept_distance(i, j) for i in all_concepts] for j in
        all_concepts])

    return utils.emd(np.array(d1), np.array(d2), distance_matrix)


@lru_cache(maxmem=MAXMEM)
def constellation_distance(C1, C2, null_concept):
    """Return the distance between two constellations in concept-space."""
    concepts_only_in_C1 = [c for c in C1 if c not in C2]
    concepts_only_in_C2 = [c for c in C2 if c not in C1]
    # If the only difference in the constellations is that some concepts
    # disappeared, then we don't need to use the emd.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        return _constellation_distance_simple(C1, C2, null_concept)
    else:
        return _constellation_distance_emd(C1, C2,
                                           concepts_only_in_C1,
                                           concepts_only_in_C2,
                                           null_concept)


# TODO Define this for cuts? need to have a cut in the null concept then
def conceptual_information(subsystem):
    """Return the conceptual information for a subsystem.

    This is the distance from the subsystem's constellation to the null
    concept."""
    return constellation_distance(constellation(subsystem), ())


# TODO document
def _null_mip(subsystem):
    """Returns a BigMip with zero phi and empty constellations.

    This is the MIP associated with a reducible subsystem."""
    return BigMip(subsystem=subsystem,
                  phi=0.0,
                  cut=subsystem.null_cut,
                  unpartitioned_constellation=[], partitioned_constellation=[])


def _single_node_mip(subsystem):
    """Returns a the BigMip of a single-node with a selfloop.

    Whether these have a nonzero |Phi| value depends on the CyPhi options.
    """
    if options.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI:
        # TODO return the actual concept
        return BigMip(
            phi=0.5,
            cut=Cut(subsystem.nodes, subsystem.nodes),
            unpartitioned_constellation=None,
            partitioned_constellation=None,
            subsystem=subsystem)
    else:
        return _null_mip(subsystem)


# TODO document
def _evaluate_cut(subsystem, partition, unpartitioned_constellation):
    # Compute forward mip.
    forward_cut = Cut(partition[0], partition[1])
    forward_cut_subsystem = Subsystem(subsystem.node_indices,
                                      subsystem.current_state,
                                      subsystem.past_state,
                                      subsystem.network,
                                      cut=forward_cut)
    forward_constellation = constellation(forward_cut_subsystem)
    forward_mip = BigMip(
        phi=constellation_distance(unpartitioned_constellation,
                                   forward_constellation,
                                   subsystem.null_concept()),
        cut=forward_cut,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=forward_constellation,
        subsystem=subsystem)
    # Compute backward mip.
    backward_cut = Cut(partition[1], partition[0])
    backward_cut_subsystem = Subsystem(subsystem.node_indices,
                                       subsystem.current_state,
                                       subsystem.past_state,
                                       subsystem.network,
                                       cut=backward_cut)
    backward_constellation = constellation(backward_cut_subsystem)
    backward_mip = BigMip(
        phi=constellation_distance(unpartitioned_constellation,
                                   backward_constellation,
                                   subsystem.null_concept()),
        cut=backward_cut,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=backward_constellation,
        subsystem=subsystem)
    # Choose minimal unidirectional cut.
    return min(forward_mip, backward_mip)


# TODO document big_mip
# @memory.cache(ignore=['subsystem'])
def _big_mip(subsystem, cache_key):
    """Return the MIP for a subsystem.

    Args:
        subsystem (Subsystem): The subsystem of the network for which to
            calculate |big_phi|.
        cache_key (str): The key to use for persistent caching with
            ``joblib.Memory``.
    """
    # Special case for single-node subsystems.
    if (len(subsystem.node_indices) == 1):
        return _single_node_mip(subsystem)

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty; or
    #   - an elementary mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null MIP.

    if not subsystem.node_indices:
        return _null_mip(subsystem)

    if subsystem.connectivity_matrix is not None:
        # Get the connectivity of just the subsystem nodes.
        submatrix_indices = np.ix_(subsystem.node_indices,
                                   subsystem.node_indices)
        cm = subsystem.network.connectivity_matrix[submatrix_indices]
        # Get the number of strongly connected components.
        num_components, _ = connected_components(csr_matrix(cm))
        if num_components > 1:
            return _null_mip(subsystem)

    # The first bipartition is the null cut (trivial bipartition), so skip it.
    bipartitions = utils.bipartition(subsystem.node_indices)[1:]
    if not bipartitions:
        return _null_mip(subsystem)

    # =========================================================================

    # Calculate the unpartitioned constellation.
    unpartitioned_constellation = constellation(subsystem)
    # Parallel loop over all partitions (use all but one CPU).
    mip_candidates = Parallel(n_jobs=(-2 if options.PARALLEL_CUT_EVALUATION
                                      else 1),
                              verbose=options.VERBOSE_PARALLEL)(
        delayed(_evaluate_cut)(subsystem,
                               partition,
                               unpartitioned_constellation)
        for partition in bipartitions)

    return min(mip_candidates)


# Wrapper so that joblib.Memory caches by the native hash
@functools.wraps(_big_mip)
def big_mip(subsystem):
    return _big_mip(subsystem, hash(subsystem))


@lru_cache(maxmem=MAXMEM)
def big_phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return big_mip(subsystem).phi


@lru_cache(maxmem=MAXMEM)
def complexes(network):
    """Return a generator for all complexes of the network.

    This includes reducible, zero-phi complexes (which are not, strictly
    speaking, complexes at all)."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return (big_mip(subsystem) for subsystem in network.subsystems())


@lru_cache(maxmem=MAXMEM)
def main_complex(network):
    """Return the main complex of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return max(complexes(network))
