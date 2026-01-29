## Preparing a bam file for `coelsch loadbam`

The recommended method for preparing data for crossover analysis with `coelsch` is to use `STARsolo` in diploid mode. This mode is available in `STAR` version 2.7.11a onwards and allows reads to be mapped to a heterozygous diploid genome, i.e. two phased haplotypes at once. In order to do this, `STAR` requires a reference genome of haplotype 1, and a vcf file containing variants that transform haplotype 1 into haplotype 2, to be supplied during genome indexing.

### Preparing a vcf file for `STARsolo` diploid mode

If you have fully assembled genomes for both haplotypes, you can use `minimap2` to align them to each other, followed by analysis with `syri` to identify variants. As part of the `coelsch` package we provide a post-processing script for filtering `syri` output, to retain variants in syntenic regions that are useful for crossover analysis.

Here is an example of how to prepare variants for analysis of gametes from an Arabidopsis Col-0 × Ler-0 F1 hybrid. First we align the Ler-0 genome to the Col-0 reference using minimap2:


```bash
minimap2 --eqx -ax "asm20" "col0.fa" "ler0.fa" | \
  samtools view -bS | \
  samtools sort > "col0_ler0.aln.bam"
samtools index "col0_ler0.aln.bam"
```

    [M::mm_idx_gen::2.446*0.99] collected minimizers
    [M::mm_idx_gen::2.852*1.28] sorted minimizers
    [M::main::2.852*1.28] loaded/built the index for 5 target sequence(s)
    [M::mm_mapopt_update::3.114*1.25] mid_occ = 76
    [M::mm_idx_stat] kmer size: 19; skip: 10; is_hpc: 0; #seq: 5
    [M::mm_idx_stat::3.283*1.24] distinct minimizers: 19682606 (95.50% are singletons); average occurrences: 1.238; average spacing: 5.473; total length: 133336906
    [M::worker_pipeline::83.352*2.44] mapped 5 sequences
    [M::main] Version: 2.28-r1209
    [M::main] CMD: /opt/share/software/packages/minimap2-v2.28/bin/minimap2 --eqx -ax asm20 col0.fa ler0.fa
    [M::main] Real time: 83.411 sec; CPU: 203.222 sec; Peak RSS: 5.067 GB


Next we run `syri` to generate a vcf file describing the rearrangements and variants between the two genomes. Make sure to use the `--hdrseq` flag to output the entire sequence of alleles rather than symbolic alleles.


```bash
syri -F B -f --hdrseq \
  --prefix "col0_ler0." \
  -c "col0_ler0.aln.bam" \
  -r "col0.fa" \
  -q "ler0.fa"
```


Then we use the `syri_vcf_to_stardiploid.py` script provided with `coelsch` to filter the variants and convert the vcf into one compatible with `STAR`.


```bash
syri_vcf_to_stardiploid.py \
  --alt-sample-name "ler0" \
  "col0_ler0.syri.vcf" \
  "col0_ler0.stardiploid.vcf"

head col0_ler0.stardiploid.vcf
```

    ##fileformat=VCFv4.3
    ##source=syri_vcf_to_stardiploid.py
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	ler0
    Chr1	12433	INS1238	G	GCTCTCT	.	PASS	MTD=syri	GT	0|1
    Chr1	13736	DEL1239	CAT	C	.	PASS	MTD=syri	GT	0|1
    Chr1	14669	SNP1240	A	T	.	PASS	MTD=syri	GT	0|1
    Chr1	17110	INS1241	A	AAG	.	PASS	MTD=syri	GT	0|1
    Chr1	18299	DEL1242	GA	G	.	PASS	MTD=syri	GT	0|1
    Chr1	21822	SNP1243	C	T	.	PASS	MTD=syri	GT	0|1
    Chr1	22723	DEL1244	CT	C	.	PASS	MTD=syri	GT	0|1


### Running `STARsolo` in diploid mode

The next step is to build the diploid genome index for `STAR`. This is done by adding the parameters `--genomeTransformVCF` and `--genomeTransformType` to the normal `STAR --runMode genomeGenerate` command. You will also have to provide a gtf file of gene annotations for the reference genome, although the use of this is not necessary for downstream `coelsch` analysis. Please read the `STAR` manual for further information and more options.


```bash
STAR \
  --runThreadN 10 \
  --runMode "genomeGenerate" \
  --genomeDir "col0_ler_star_index" \
  --genomeFastaFiles "col0.fa" \
  --sjdbGTFfile "col0.gtf" \
  --genomeTransformVCF "col0_ler0.stardiploid.vcf" \
  --genomeTransformType "Diploid" \
  --genomeSAindexNbases 12
```

    Oct 29 13:39:16 ..... started STAR run
    Oct 29 13:39:16 ... starting to generate Genome files
    Oct 29 13:39:18 ..... processing annotations GTF
    Oct 29 13:39:21 ... starting to sort Suffix Array. This may take a long time...
    Oct 29 13:39:22 ... sorting Suffix Array chunks and saving them to disk...
    Oct 29 13:46:41 ... loading chunks from disk, packing SA...
    Oct 29 13:46:47 ... finished generating suffix array
    Oct 29 13:46:47 ... generating Suffix Array index
    Oct 29 13:47:01 ... completed Suffix Array index
    Oct 29 13:47:01 ..... inserting junctions into the genome indices
    Oct 29 13:48:25 ... writing Genome to disk ...
    Oct 29 13:48:26 ... writing Suffix Array to disk ...
    Oct 29 13:48:27 ... writing SAindex to disk
    Oct 29 13:48:27 ..... finished successfully


Finally, we can now map our single cell/nucleus sequencing reads to the diploid genome! The commands that you use to do this will depend slightly on the type of data that you have. For example, for 10x Genomics 3' scRNA sequencing data, which has both cell barcodes and unique molecular identifiers, as well as spliced reads, we provide a command like this:


```bash
STAR \
  --runThreadN 12 \
  --genomeDir "col0_ler_star_index" \
  --readFilesIn "sn_gametes.2.fastq.gz" "sn_gametes.1.fastq.gz" \
  --readFilesCommand "zcat" \
  --soloType "CB_samTagOut" \
  --soloCBmatchWLtype "1MM" \
  --soloCBwhitelist "3M-february-2018.txt" \
  --soloUMIlen 12 \
  --soloUMIdedup "NoDedup" \
  --outFilterMultimapNmax 2 \
  --outFilterIntronMotifs "RemoveNoncanonical" \
  --outFilterMismatchNmax 4 \
  --outSAMtype "BAM SortedByCoordinate" \
  --outSAMattributes "NH HI AS nM NM CB UR ha" \
  --genomeTransformOutput "SAM"
```

    Oct 29 14:01:11 ..... started STAR run
    Oct 29 14:01:12 ..... loading genome
    Oct 29 14:01:15 ..... started mapping
    Oct 29 14:06:17 ..... finished mapping
    Oct 29 14:06:17 ..... started sorting BAM
    Oct 29 14:06:49 ..... finished successfully


__NB: the current version of STARsolo + diploid mode (version 2.7.11b) has a bug, which prevents correct deduplication of UMIs for reads that map to haplotype 2.__ See [STAR issue #2112](https://github.com/alexdobin/STAR/issues/2112) for details. Until this is resolved, it is recommended to switch off UMI deduplication with STAR using `--soloUMIdedup "NoDedup"`, and perform deduplication with `coelsch loadbam --umi-collapse-method="directional"` instead.

If your data is from a 10x single cell/nucleus ATAC sequencing experiment, it is still possible to use `STARsolo` diploid mode to perform alignment, but some adjustments have to be made. The main one is to switch of splice alignment using `--alignIntronMax 1` and `--alignMatesGapMax 500`. There is further information on how best to use `STAR` with ATAC-seq data on the `STAR` GitHub issue tracker.

The output of the `STARsolo` command is a bam file, in which each read has a `CB` tag that defines the cell barcode, a `UR` tag that defines the UMI, and a `ha` tag that identifies which of the two haplotypes it aligns best to. This is then the input for the `coelsch` analysis...
