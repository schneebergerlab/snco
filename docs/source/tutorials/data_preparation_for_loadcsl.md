## Preparing a bam file for `coelsch loadcsl`

If you prefer to work with SNPs rather than marker reads, then `coelsch` also provides a load command for reading data from the output of [`cellsnp-lite`](https://github.com/single-cell-genetics/cellsnp-lite). This is a tool that uses a bam file (aligned to the reference genome i.e. haplotype 1) with cell barcode tags, along with a vcf of known variants (in haplotype 2) to call SNPs for each individual cell barcode. In order to produce a high quality vcf file of variants that are syntenic and co-linear, we still recommend to use the same `minimap2 + syri + syri_vcr_to_stardiploid.py` method described in the [Data preparation for `loadbam`](./data_preparation_for_loadbam.md) page.


```bash
cellsnp-lite \
  -s sn_gametes.bam \
  -b 3M-february-2018.txt \
  -R col0_ler0.stardiploid.vcf \
  -p 12 \
  --minCount 0 \
  -O sn_gametes_csl
```

    [I::main] start time: 2024-10-29 16:57:03
    [W::check_args] Max depth set to maximum value (2147483647)
    [W::check_args] Max pileup set to maximum value (2147483647)
    [I::main] loading the VCF file for given SNPs ...
    [I::main] fetching 818731 candidate variants ...
    [I::main] mode 1a: fetch given SNPs in 1847 single cells.
    [I::main] All Done!
    [I::main] Version: 1.2.3 (htslib 1.21)
    [I::main] CMD: cellsnp-lite -s sn_gametes.bam -b 3M-february-2018.txt -R col0_ler0.stardiploid.vcf -p 12 --minCount 0 -O sn_gametes_csl
    [I::main] end time: 2024-10-29 16:57:24
    [I::main] time spent: 21 seconds.


The output of `cellsnp-lite` is a set of three sparse matrices (in matrix market format), and a list of barcodes and variants:


```bash
ls sn_gametes_csl
```

    cellSNP.base.vcf     cellSNP.tag.AD.mtx   cellSNP.tag.OTH.mtx
    cellSNP.samples.tsv  cellSNP.tag.DP.mtx


These output files can be used directly with `coelsch loadcsl`.
