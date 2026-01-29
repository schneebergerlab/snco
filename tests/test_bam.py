import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def make_mock_alignment():
    def _factory(cb='ATAGATG', ub='AGTAC', ha=1, start=10_000, **kwargs):
        aln = MagicMock()
        aln.reference_start = start
        aln.is_secondary = False
        aln.is_supplementary = False
        aln.is_duplicate = False
        aln.is_paired = False
        aln.get_tag.side_effect = lambda tag: {
            'CB': cb,
            'UB': ub,
            'ha': ha
        }[tag]
        if kwargs:
            for attr, val in kwargs.items():
                setattr(aln, attr, val)
        return aln
    return _factory


@pytest.fixture
def mock_bamfile(make_mock_alignment):
    bam = MagicMock()
    bam.references = ['Chr1']
    bam.get_reference_length.return_value = 1_000_000
    bam.fetch.return_value = [
        make_mock_alignment(),
        make_mock_alignment(start=10_010),
        make_mock_alignment(start=10_011),
        make_mock_alignment(ub='AGAAC'), # umi error
        make_mock_alignment(ub='GGTCC'), # unique umi
        make_mock_alignment(cb='ATAGATC', ub='GGTCC'), # cb mismatch
        make_mock_alignment(
            cb='GATCCGC', ub='GACGG', is_secondary=True # secondary, ignored
        ),
    ]
    return bam


@pytest.fixture
def patched_alignment_file(mock_bamfile):
    with patch('coelsch.load.loadbam.bam.pysam.AlignmentFile', return_value=mock_bamfile) as patched:
        yield patched


@pytest.fixture
def reader_factory(patched_alignment_file):
    from coelsch.load.loadbam.bam import BAMHaplotypeIntervalReader
    from coelsch.load.barcodes.cb import CellBarcodeWhitelist

    def _create(whitelist, cb_correction_method='exact', umi_collapse_method='directional'):
        return BAMHaplotypeIntervalReader(
            "dummy.bam",
            bin_size=25_000,
            umi_collapse_method=umi_collapse_method,
            cb_whitelist=CellBarcodeWhitelist(
                whitelist,
                correction_method=cb_correction_method
            )
        )
    return _create


full_whitelist = ['ATAGATG', 'ATAGATC', 'GATCCGC']
whitelist_no_mm = ['ATAGATG', 'GATCCGC']


def test_reader_creation(reader_factory):
    reader = reader_factory(None)
    assert reader.chrom_sizes == {'Chr1': 1_000_000}
    assert reader.nbins == {'Chr1': 40}


@pytest.mark.parametrize(
    'whitelist, cb_correction_method, umi_collapse_method, expected_counts', 
    [
        (whitelist_no_mm, '1mm', 'directional', {'ATAGATG': {0: 2}}),
        (full_whitelist, 'exact', 'directional', {'ATAGATG': {0: 2}, 'ATAGATC': {0: 1}}),
        (full_whitelist, 'exact', 'exact', {'ATAGATG': {0: 3}, 'ATAGATC': {0: 1}}),
        (full_whitelist, 'exact', None, {'ATAGATG': {0: 5}, 'ATAGATC': {0: 1}})
    ]
)
def test_reader_counts(reader_factory, whitelist,
                       cb_correction_method,
                       umi_collapse_method,
                       expected_counts):
    reader = reader_factory(
        whitelist,
        cb_correction_method=cb_correction_method,
        umi_collapse_method=umi_collapse_method
    )
    counts = reader.fetch_interval_counts("Chr1", 0)
    assert counts.counts == expected_counts


@pytest.fixture
def multi_hap_mock_alignment():
    def _factory(cb='ATAGATG', ub='AGTAC', ha='col0,ler', start=0, **kwargs):
        aln = MagicMock()
        aln.reference_start = start
        aln.is_secondary = False
        aln.is_supplementary = False
        aln.is_duplicate = False
        aln.is_paired = False
        aln.get_tag.side_effect = lambda tag: {
            'CB': cb,
            'UB': ub,
            'ha': ha
        }[tag]
        for attr, val in kwargs.items():
            setattr(aln, attr, val)
        return aln
    return _factory


def test_multi_haplotype_validated(multi_hap_mock_alignment):
    from coelsch.load.loadbam.bam import BAMHaplotypeIntervalReader, MultiHaplotypeValidator

    aln_valid = multi_hap_mock_alignment()  # allowed set
    aln_equal = multi_hap_mock_alignment(ha='col0,ler,db1')  # equals validator → should be skipped
    alns_disallowed = [multi_hap_mock_alignment(cb='GATCCGC', ha='col0'),
                       multi_hap_mock_alignment(cb='GATCCGC', ha='ler')]

    mock_bam = MagicMock()
    mock_bam.references = ['Chr1']
    mock_bam.get_reference_length.return_value = 100_000
    mock_bam.fetch.return_value = [aln_valid, aln_equal] + alns_disallowed

    with patch('coelsch.load.loadbam.bam.pysam.AlignmentFile', return_value=mock_bam):
        reader = BAMHaplotypeIntervalReader(
            'dummy.bam',
            hap_tag_type='multi_haplotype',
            umi_collapse_method='directional',
            allowed_haplotypes=['col0', 'ler', 'db1'],
            cb_whitelist=None
        )
        counts = reader.fetch_interval_counts("Chr1", 0)
        assert counts.counts == {
            'ATAGATG': {frozenset({'col0', 'ler'}): 1}
        }