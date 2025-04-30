import tempfile
import pytest
import numpy as np
from snco.records import MarkerRecords, PredictionRecords, NestedData
from snco.bam import IntervalMarkerCounts


@pytest.fixture
def chrom_sizes():
    return {"Chr1": 1_000_000, "Chr2": 1_200_000}


@pytest.fixture
def marker_records(chrom_sizes):
    return MarkerRecords(chrom_sizes=chrom_sizes, bin_size=25_000)


@pytest.fixture
def pred_records(chrom_sizes):
    return PredictionRecords(chrom_sizes=chrom_sizes, bin_size=25_000)


@pytest.fixture
def simple_nested_data():
    return NestedData(
        levels=('cb', 'chrom'),
        dtype=(int,),
        data={'ATATGT': {'Chr1': 1, 'Chr2': 2}, 'GACGTT': {'Chr1': 3, 'Chr3': 4}}
    )

@pytest.fixture
def interval_counts():
    return IntervalMarkerCounts(
        chrom='Chr1',
        bin_idx=10,
        counts={'ATATGT': {0: 10}}
    )

def test_marker_record_array_creation(marker_records):
    arr = marker_records['ATATGT', 'Chr1']
    assert arr.shape == (40, 2)
    with pytest.raises(ValueError):
        arr = marker_records['ATATGT', 'ChrX']


def test_pred_record_array_creation(pred_records):
    arr = pred_records['ATATGT', 'Chr1']
    assert arr.shape == (40,)


def test_marker_record_array_assignment_and_retrieval(marker_records):
    marker_records['ATATGT', 'Chr1', 1, 0] = 2
    val = marker_records['ATATGT', "Chr1", 1, 0]
    assert val == 2


def test_add_metadata(marker_records, simple_nested_data):
    marker_records.add_metadata(test=simple_nested_data)
    assert 'test' in marker_records.metadata
    assert marker_records.metadata['test'] == simple_nested_data


def test_marker_total_count(marker_records):
    marker_records['ATATGT', 'Chr1', 1, 0] = 2
    marker_records['ATATGT', 'Chr1', 10, 1] = 2
    assert marker_records.total_marker_count('ATATGT') == 4


def test_add_cb_suffix(marker_records):
    marker_records['ATATGT', 'Chr1', 1, 0] = 2
    updated = marker_records.add_cb_suffix('1', inplace=False)
    assert 'ATATGT_1' in updated
    assert 'ATATGT' in marker_records


def test_filter(marker_records):
    marker_records['ATATGT', 'Chr1', 0, 0] = 1
    marker_records['CCCCGT', 'Chr1', 0, 0] = 2
    filtered = marker_records.filter(cb_whitelist=['ATATGT'], inplace=False)
    assert 'ATATGT' in filtered
    assert 'CCCCGT' not in filtered
    assert 'CCCCGT'  in marker_records


def test_merge_same_class(marker_records, chrom_sizes):
    other = MarkerRecords(chrom_sizes, bin_size=25_000)
    other['ATATGT', 'Chr2', 1, 0] = 4
    merged = marker_records.merge(other, inplace=False)
    assert 'ATATGT' in merged
    assert 'ATATGT' not in marker_records


def test_invalid_merge_mismatch_class(marker_records, pred_records):
    with pytest.raises(ValueError):
        marker_records.merge(pred_records)


def test_to_from_json(marker_records):
    marker_records['ATATGT', 'Chr1', 0, 0] = 1
    tmp_fh, tmp_fn = tempfile.mkstemp()
    marker_records.write_json(tmp_fn, precision=2)
    loaded = MarkerRecords.read_json(tmp_fn)
    assert 'ATATGT' in loaded
    assert loaded.chrom_sizes == marker_records.chrom_sizes
    assert loaded.bin_size == marker_records.bin_size
    assert loaded['ATATGT', 'Chr1', 0, 0] == 1


def test_iadd_marker_records(marker_records, chrom_sizes):
    other = MarkerRecords(chrom_sizes, bin_size=25_000)
    marker_records['ATATGT', 'Chr2', 1, 0] = 42
    other['ATATGT', 'Chr2', 1, 0] = 4
    marker_records += other
    assert marker_records['ATATGT', 'Chr2', 1, 0] == 46


def test_iadd_interval_marker_counts(marker_records, interval_counts):
    marker_records += interval_counts
    assert marker_records["ATATGT", "Chr1", 10, 0] == 10