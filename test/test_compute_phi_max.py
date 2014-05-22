#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from itertools import chain

from cyphi.models import Mice
from cyphi.utils import phi_eq, nodes2indices
import cyphi.compute as compute

import example_networks


# Expected results {{{
# ====================

subsystem = example_networks.s()
directions = ['past', 'future']


def indices2nodes(indices):
    return tuple(subsystem.network.nodes[index] for index in indices)


expected_purview_indices = {
    'past': {
        (1,): (2,),
        (2,): (0, 1),
        (0, 1): (1, 2),
        (0, 1, 2): (0, 1, 2)},
    'future': {
        (1,): (0,),
        (2,): (1,),
        (0, 1): (2,),
        (0, 1, 2): (0, 1, 2)}
}
expected_purviews = {
    direction: {
        mechanism_indices: indices2nodes(purview_indices) for
        mechanism_indices, purview_indices in
        expected_purview_indices[direction].items()
    } for direction in directions
}
expected_mips = {
    direction: {
        mechanism_indices: subsystem.find_mip(direction,
                                              indices2nodes(mechanism_indices),
                                              purview)
        for mechanism_indices, purview in expected_purviews[direction].items()
    } for direction in directions
}

expected_mice = {
    direction: [
        Mice(direction=direction,
             mechanism=mechanism_indices,
             purview=expected_purview_indices[direction][mechanism_indices],
             repertoire=mip.unpartitioned_repertoire,
             mip=mip,
             phi=mip.phi)
        for mechanism_indices, mip in expected_mips[direction].items()
    ] for direction in directions
}

# }}}
# `find_mice` tests {{{
# =====================

mice_scenarios = [
    [(direction, mice) for mice in expected_mice[direction]]
    for direction in directions
]
mice_scenarios = list(chain(*mice_scenarios))
mice_parameter_string = "direction,expected"


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_find_mice(direction, expected):
    result = compute.find_mice(subsystem, direction,
                               expected.mechanism)

    print('Expected:\n', expected, '\n')
    print('Result:\n', result)

    assert (compute.find_mice(subsystem, direction, expected.mechanism) ==
            expected)


def test_find_mice_empty(s):
    expected = [
        Mice(direction=direction,
             mechanism=(),
             purview=nodes2indices(s.nodes),
             repertoire=None,
             mip=s._null_mip(direction, (), s.nodes),
             phi=0)
        for direction in directions]
    assert all(compute.find_mice(s, mice.direction,
                                 mice.mechanism) == mice for
               mice in expected)


# Test input validation
def test_find_mice_validation_bad_direction(s):
    mechanism = tuple([0])
    with pytest.raises(ValueError):
        compute.find_mice(s, 'doge', mechanism)


def test_find_mice_validation_nonnode(s):
    with pytest.raises(ValueError):
        compute.find_mice(s, 'past', (0, 1))


def test_find_mice_validation_noniterable(s):
    with pytest.raises(ValueError):
        compute.find_mice(s, 'past', tuple([0]))

# }}}
# `phi_max` tests {{{
# ===================


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_core_cause_or_effect(direction, expected):
    if direction == 'past':
        core_ce = compute.core_cause
    elif direction == 'future':
        core_ce = compute.core_effect
    else:
        raise ValueError("Direction must be 'past' or 'future'")
    answer = core_ce(subsystem, expected.mechanism)
    print('Answer:\n', answer, '\n')
    print('Expected:\n', expected)
    assert answer == expected


phi_max_scenarios = [
    (past.mechanism, min(past.phi, future.phi))
    for past, future in zip(expected_mice['past'], expected_mice['future'])
]


@pytest.mark.parametrize('mechanism, expected_phi_max', phi_max_scenarios)
def test_phi_max(s, expected_phi_max, mechanism):
    assert phi_eq(compute.phi_max(s, mechanism), expected_phi_max)

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
