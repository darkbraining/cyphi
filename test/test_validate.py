#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from cyphi.models import Cut
from cyphi import validate



def test_validate_nodelist_noniterable():
    with pytest.raises(ValueError):
        validate.nodelist(2, "it's a doge")


def test_validate_nodelist_nonnode():
    with pytest.raises(ValueError):
        validate.nodelist([0, 1, 2], 'invest in dogecoin!')


def test_validate_direction():
    with pytest.raises(ValueError):
        validate.direction("dogeeeee")


def test_validate_cut_singleton_input(s):
    should_silently_convert = validate.cut(s, (0, 1))
    print(should_silently_convert)
    should_silently_convert = validate.cut(s, (0, (1, 1)))
    print(should_silently_convert)


def test_validate_cut_single_node(s):
    validated = validate.cut(s, (0, (1, 2)))
    assert validated == Cut((0,), (1, 2))


def test_validate_cut_list_input(s):
    validated = validate.cut(s, ([0], [1, 2]))
    assert validated == Cut((0,), (1, 2))


def test_validate_cut(s):
    validated = validate.cut(s, ((0,), (1, 2)))
    assert validated == Cut((0,), (1, 2))
