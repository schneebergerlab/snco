from copy import deepcopy
import pytest

from snco.records import MarkerRecords, NestedData
from snco.records.groupby import RecordsGroupyBy, dummy_grouper, genotype_grouper

@pytest.fixture
def marker_records():
    chrom_sizes = {'Chr1': 1_000_000}
    bin_size = 250_000

    markers = MarkerRecords(chrom_sizes=chrom_sizes, bin_size=bin_size)
    markers['ATATGTG', 'Chr1', 0, 0] = 1
    markers['CAGTTTG', 'Chr1', 0, 0] = 2
    markers['ACGGTTG', 'Chr1', 0, 0] = 3

    markers.add_metadata(
        genotypes=NestedData(
            levels=('cb',),
            dtype=str,
            data={
                'ATATGTG': 'parent1:parent2',
                'CAGTTTG': 'parent1:parent2',
                'ACGGTTG': 'parent1:parent3',
            }
        )
    )

    return markers


def test_dummy_grouper(marker_records):
    grouper = dummy_grouper(marker_records, dummy_name='ungrouped')
    assert grouper('ATATGTG') == 'ungrouped'


def test_genotype_grouper(marker_records):
    grouper = genotype_grouper(marker_records)
    assert grouper('ATATGTG') == 'parent1:parent2'
    assert grouper('CAGTTTG') == 'parent1:parent2'
    assert grouper('ACGGTTG') == 'parent1:parent3'


def test_genotype_grouper_missing_barcode_key_raises(marker_records):
    marker_records = deepcopy(marker_records)
    marker_records.metadata['genotypes'].pop('ATATGTG')
    grouper = genotype_grouper(marker_records)
    with pytest.raises(KeyError):
        grouper('ATATGTG')


def test_groupby_genotype(marker_records):
    gb = RecordsGroupyBy(marker_records, 'genotype')
    assert len(gb) == 2
    group_keys = [g for g, _ in gb]
    assert set(group_keys) == {'parent1:parent2', 'parent1:parent3'}


def test_grouper_getitem(marker_records):
    gb = RecordsGroupyBy(marker_records, 'genotype')
    group1 = gb['parent1:parent2']
    assert set(group1.barcodes) == {'ATATGTG', 'CAGTTTG'}


def test_groupby_dict(marker_records):
    mapping = {'ATATGTG': 'grp1', 'CAGTTTG': 'grp2', 'ACGGTTG': 'grp1'}
    gb = RecordsGroupyBy(marker_records, mapping)
    assert len(gb) == 2
    group1 = gb['grp1']
    assert set(group1.barcodes) == {'ATATGTG', 'ACGGTTG'}


def test_groupby_callable(marker_records):
    def custom(cb, records, genotypes):
        # test signature is working
        assert records == marker_records
        assert genotypes == marker_records.metadata['genotypes']
        return cb[0] # group by first base of barcode
    gb = RecordsGroupyBy(marker_records, custom)
    assert len(gb) == 2
    group1 = gb['A']
    assert set(group1.barcodes) == {'ATATGTG', 'ACGGTTG'}


def test_groupby_apply(marker_records):
    gb = RecordsGroupyBy(marker_records, 'genotype')

    def scale_up(group, factor):
        for *_, arr in group.deep_items():
            arr *= factor
        return group

    out = gb.apply(scale_up, factor=2)
    assert out['ATATGTG', 'Chr1', 0, 0] == 2
    assert out['CAGTTTG', 'Chr1', 0, 0] == 4
    assert out['ACGGTTG', 'Chr1', 0, 0] == 6
