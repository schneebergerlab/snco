import pytest
import numpy as np
from coelsch.records import NestedData, NestedDataArray

@pytest.fixture
def simple_nested_data():
    return NestedData(
        levels=('cb', 'chrom'),
        dtype=(int,),
        data={'ATATGT': {'Chr1': 1, 'Chr2': 2}, 'GACGTT': {'Chr1': 3, 'Chr3': 4}}
    )

@pytest.fixture
def three_level_nested_data():
    return NestedData(
        levels=('geno', 'cb', 'chrom'),
        dtype=(int,),
        data={'geno1': {'ATATAG': {'Chr1': 3}, 'CCGAGT': {'Chr1': 1}},
              'geno2': {'ATATAG': {'Chr1': 2}}}
    )


@pytest.fixture
def simple_nested_data_array():
    return NestedDataArray(
        levels=('cb', 'chrom'),
        data={'ATATGT': {'Chr1': np.arange(10), 'Chr2': np.arange(20)},
              'GACGTT': {'Chr1': np.arange(10), 'Chr3': np.arange(15)}}
    )


def test_initialization_valid():
    nd = NestedData(levels='cb', dtype=float)
    assert nd.levels == ('cb',)
    assert nd.dtype == (float,)
    assert nd.nlevels == 1
    assert nd._data == {}


def test_initialization_invalid_levels():
    with pytest.raises(ValueError):
        NestedData(levels=tuple(), dtype=(int,))


def test_initialization_invalid_dtype():
    with pytest.raises(ValueError):
        NestedData(levels=('cb',), dtype=('not_a_type',))


def test_getitem_leaf(simple_nested_data):
    assert simple_nested_data['ATATGT', 'Chr1'] == 1


def test_getitem_sublevel(simple_nested_data):
    selection = simple_nested_data['ATATGT']
    assert isinstance(selection, NestedData)
    assert selection.levels == ('chrom',)
    assert 'Chr1' in selection
    assert selection['Chr1'] == 1


def test_get_level_keys(simple_nested_data):
    keys = simple_nested_data.get_level_keys('chrom')
    assert set(keys) == {'Chr1', 'Chr2', 'Chr3'}


def test_getitem_slice(three_level_nested_data):
    selection = three_level_nested_data[:, 'ATATAG']
    assert selection.levels == ('geno', 'chrom')
    assert set(selection.keys()) == {'geno1', 'geno2'}
    assert set(selection.get_level_keys(level='chrom')) == {'Chr1'}


def test_getitem_list(three_level_nested_data):
    selection = three_level_nested_data[:, ['ATATAG']]
    assert selection.levels == ('geno', 'cb', 'chrom')
    assert set(selection.get_level_keys(level='cb')) == {'ATATAG'}


def test_getitem_ellipsis(three_level_nested_data):
    selection = three_level_nested_data[..., 'Chr1']
    assert selection.levels == ('geno', 'cb')
    assert set(selection.keys()) == {'geno1', 'geno2'}
    assert set(selection.get_level_keys(level='cb')) == {'ATATAG', 'CCGAGT'}
    assert selection['geno1', 'ATATAG'] == 3


def test_invalid_getitem_too_many_levels(simple_nested_data):
    # Test indexing beyond available levels
    with pytest.raises(KeyError):
        # simple case only has cb, chrom
        simple_nested_data['ATATGT', 'Chr1', 'extra_level']


def test_invalid_getitem_nonexistent_key(simple_nested_data):
    # Test indexing with a key that doesn't exist
    with pytest.raises(KeyError):
        simple_nested_data['ATATGT', 'ChrX']


def test_setitem_and_getitem_leaf(simple_nested_data):
    simple_nested_data['ACCCCT', 'Chr1'] = 10
    assert simple_nested_data['ACCCCT', 'Chr1'] == 10


def test_setitem_and_getitem_sublevel(simple_nested_data):
    simple_nested_data['CGGAGT'] = {'Chr1': 2}
    assert simple_nested_data['CGGAGT', 'Chr1'] == 2


def test_invalid_setitem_wrong_levels(simple_nested_data):
    # too many levels for this object with nlevels == 2
    with pytest.raises(ValueError):
        simple_nested_data['CGGAGT'] = {'Chr1': {'extra_level': 2}}


def test_getitem_slice_data_array(simple_nested_data_array):
    selection = simple_nested_data_array['ATATGT', 'Chr1', :10]
    assert isinstance(selection, np.ndarray)
    assert selection.shape == (10,)


def test_getitem_nested_slice_data_array(simple_nested_data_array):
    selection = simple_nested_data_array[:, 'Chr1', :10]
    assert isinstance(selection, NestedDataArray)
    assert isinstance(selection['ATATGT'], np.ndarray)
    assert selection['ATATGT'].shape == (10,)


def test_setitem_and_getitem_data_array(simple_nested_data_array):
    simple_nested_data_array[:, 'Chr1', 4] = 1
    assert simple_nested_data_array['ATATGT', 'Chr1', 4] == 1
    assert simple_nested_data_array['GACGTT', 'Chr1', 4] == 1


def test_new_like(simple_nested_data):
    nd_new = NestedData.new_like(simple_nested_data)
    assert simple_nested_data.levels == nd_new.levels
    assert nd_new._data == {}


def test_copy_creates_independent_object(simple_nested_data):
    nd_copy = simple_nested_data.copy()
    nd_copy['ATATGT', 'Chr1'] = 99
    assert simple_nested_data['ATATGT', 'Chr1'] == 1  # Original unchanged


def test_filter(simple_nested_data):
    filtered = simple_nested_data.filter(['ATATGT'], level='cb', inplace=False)
    assert set(filtered.keys()) == {'ATATGT'}
    assert 'GACGTT' not in filtered


def test_transpose_levels(simple_nested_data):
    transposed = simple_nested_data.transpose_levels(('chrom', 'cb'))
    assert transposed.levels == ('chrom', 'cb')
    assert set(transposed.keys()) == {'Chr1', 'Chr2', 'Chr3'}


def test_json_serialization_and_deserialization(simple_nested_data):
    json_obj = simple_nested_data.to_json()
    reloaded = NestedData.from_json(json_obj)
    assert reloaded._data == simple_nested_data._data
    assert reloaded.levels == simple_nested_data.levels


def _nested_dicts_equal(d1, d2):
    if d1.keys() != d2.keys():
        return False
    for key in d1:
        val1 = d1[key]
        val2 = d2[key]
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not _nested_dicts_equal(val1, val2):
                return False
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        else:
            assert False, "dicts dont match"
    return True


def test_json_serialization_and_deserialization_data_array_full(simple_nested_data_array):
    json_obj = simple_nested_data_array.to_json(encode_method='full')
    reloaded = NestedDataArray.from_json(json_obj)
    assert _nested_dicts_equal(reloaded._data, simple_nested_data_array._data)
    assert reloaded.levels == simple_nested_data_array.levels


def test_json_serialization_and_deserialization_data_array_sparse(simple_nested_data_array):
    json_obj = simple_nested_data_array.to_json(encode_method='sparse')
    reloaded = NestedDataArray.from_json(json_obj)
    assert _nested_dicts_equal(reloaded._data, simple_nested_data_array._data)
    assert reloaded.levels == simple_nested_data_array.levels


def test_update():
    d1 = NestedData(levels=('cb',), dtype=(int,), data={'ATATGT': 1})
    d2 = NestedData(levels=('cb',), dtype=(int,), data={'ATATGT': 2})
    d1.update(d2)
    assert d1['ATATGT'] == 2


def test_add_operator():
    d1 = NestedData(levels=('cb',), dtype=(int,), data={'ATATGT': 1})
    d2 = NestedData(levels=('cb',), dtype=(int,), data={'ATATGT': 2})
    result = d1 + d2
    assert result['ATATGT'] == 3


def test_subtract_operator():
    d1 = NestedData(levels=('cb',), dtype=(int,), data={'ATATGT': 1})
    d2 = NestedData(levels=('cb',), dtype=(int,), data={'ATATGT': 2})
    result = d2 - d1
    assert result['ATATGT'] == 1


def test_add_operator_data_array():
    d1 = NestedDataArray(levels=('cb',), data={'ATATGT': np.ones(10)})
    d2 = NestedDataArray(levels=('cb',), data={'ATATGT': np.ones(10)})
    result = d2 + d1
    assert (result['ATATGT'] == 2).all()


def test_stack_values(simple_nested_data_array):
    selection = simple_nested_data_array[..., 'Chr1']
    stacked = selection.stack_values()
    assert stacked.shape == (2, 10)


def test_invalid_stack_values(simple_nested_data_array):
    with pytest.raises(ValueError):
        # fails because different chromosomes have different array lengths
        simple_nested_data_array.stack_values()
