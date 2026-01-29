import pytest
from collections import Counter, defaultdict

# Import the function under test
from coelsch.load.barcodes.umi import umi_dedup_directional
from coelsch.load.barcodes.cb import BarcodeCorrector, BarcodeValidator


def test_barcode_validator():
    bv = BarcodeValidator(allow_ns=False, allow_homopolymers=False)
    assert bv.is_valid('AAGTGTCC')
    assert not bv.is_valid('AAAAAA')
    assert not bv.is_valid('AAGTGNNNN')


def test_barcode_validator_off():
    bv = BarcodeValidator(allow_ns=True, allow_homopolymers=True)
    assert bv.is_valid('AAGTGTCC')
    assert bv.is_valid('AAAAAA')
    assert bv.is_valid('AAGTGNNNN')


def test_barcode_validator_barcode_length():
    bv = BarcodeValidator(allow_ns=False, allow_homopolymers=False)
    bv.check_uniform_length(['AGAGTT', 'GGTGTA', 'GTGTAG'])
    with pytest.raises(ValueError):
        bv.check_uniform_length(['AGAGTT', 'GGT'])


def test_barcode_corrector():
    bc = BarcodeCorrector({'AGAGTT', 'GGTGTA', 'GTGTAG', 'GTGTGA'})
    assert bc.correct('AGAGTT') == 'AGAGTT' # no correction
    assert bc.correct('AGAGTG') == 'AGAGTT' # 1MM correction
    assert bc.correct('GTGTGG') is None # within 1MM of two barcodes
    

def test_dedup_no_haplotype_merging():
    input_counts = {
        "AAAA": 10,
        "AAAT": 4,   # Should merge into AAAA
        "CCCC": 8,
        "CCCA": 3,   # Should merge into CCCC
    }
    result = umi_dedup_directional(input_counts, has_haplotype=False)
    assert result == {'AAAA': 14, 'CCCC': 11}


def test_dedup_no_merge():
    input_counts = {
        "AAAA": 5,
        "AAAT": 4,  # Count more than 5 / 2 + 1, should not merge
    }
    result = umi_dedup_directional(input_counts, has_haplotype=False)
    assert result == {'AAAA': 5, 'AAAT': 4}


def test_dedup_with_haplotype_merging():
    input_counts = {
        "AAAA": Counter({0: 9, 1: 1}),   # total: 10
        "AAAT": Counter({0: 4}),   # total: 4 -> should merge into AAAA
        "CCCC": Counter({1: 8}),         # total: 8
        "CCCA": Counter({1: 3})          # total: 3 -> should merge into CCCC
    }
    result = umi_dedup_directional(input_counts, has_haplotype=True)
    assert result == {
        'AAAA': Counter({0: 13, 1: 1}),
        'CCCC': Counter({1: 11})
    }

def test_dedup_with_haplotype_no_merge():
    input_counts = {
        "AAAA": Counter({0: 5}),  # total: 5
        "AAAT": Counter({0: 4})   # total: 4, not mergeable
    }
    result = umi_dedup_directional(input_counts, has_haplotype=True)
    assert result == {
        "AAAA": Counter({0: 5}),
        "AAAT": Counter({0: 4})
    }


def test_empty_input():
    result = umi_dedup_directional({}, has_haplotype=False)
    assert result == {}


def test_single_umi():
    result = umi_dedup_directional({"ACGT": 10}, has_haplotype=False)
    assert result == {"ACGT": 10}
