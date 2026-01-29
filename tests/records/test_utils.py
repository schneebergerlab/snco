import pytest
import numpy as np

from coelsch.records.utils import (
    validate_data,
    array_encoder_full,
    array_encoder_sparse,
    array_decoder_full,
    array_decoder_sparse,
    deep_operator
)


def test_validate_data_valid():
    nested = {'a': {'b': np.array([1.0, 2.0])}}
    validate_data(nested, expected_depth=2, expected_dtype=np.ndarray)


def test_validate_data_wrong_depth():
    with pytest.raises(ValueError):
        validate_data({'a': np.array([1.0])}, expected_depth=2, expected_dtype=np.ndarray)


def test_validate_data_wrong_key_type():
    with pytest.raises(ValueError):
        validate_data({1: {'b': np.array([1.0])}}, expected_depth=2, expected_dtype=np.ndarray)


def test_validate_data_wrong_value_type():
    with pytest.raises(ValueError):
        validate_data({'a': {'b': 123}}, expected_depth=2, expected_dtype=np.ndarray)


def test_array_encoder_full_and_decode():
    arr = np.array([[1.23456, 2.34567, 0.0]])
    encoded = array_encoder_full(arr, precision=2)
    assert len(encoded['data']) == 3
    decoded = array_decoder_full(encoded)
    assert np.allclose(decoded, [[1.23, 2.35, 0.0]])


def test_array_encoder_sparse_and_decode():
    arr = np.array([[0.0, 3.14159, 0.0], [0.0, 0.0, 2.71828]])
    encoded = array_encoder_sparse(arr, precision=3)
    assert len(encoded['data']) == 2
    assert encoded['data'][0] == [1, 5]
    assert np.allclose(encoded['data'][1], [3.142, 2.718])
    decoded = array_decoder_sparse(encoded)
    expected = np.array([[0.0, 3.142, 0.0], [0.0, 0.0, 2.718]])
    assert np.allclose(decoded, expected)


def test_deep_operator_overwrite():
    d1 = {'x': {'y': 1.0}}
    d2 = {'x': {'y': 2.0}}
    deep_operator(d1, d2, method='overwrite')
    assert d1['x']['y'] == 2.0


def test_deep_operator_add():
    d1 = {'x': {'y': 1.0}}
    d2 = {'x': {'y': 2.0}}
    deep_operator(d1, d2, method='add')
    assert d1['x']['y'] == 3.0


def test_deep_operator_subtract():
    d1 = {'x': {'y': 5.0}}
    d2 = {'x': {'y': 3.0}}
    deep_operator(d1, d2, method='subtract')
    assert d1['x']['y'] == 2.0


def test_deep_operator_overwrite_ignore_nan_array():
    d1 = {'x': {'y': np.array([1.0, 2.0])}}
    d2 = {'x': {'y': np.array([np.nan, 5.0])}}
    deep_operator(d1, d2, method='overwrite_ignore_nan')
    assert np.allclose(d1['x']['y'], [1.0, 5.0])


def test_deep_operator_unsupported_method():
    with pytest.raises(NotImplementedError):
        deep_operator({}, {}, method='multiply')
