def read_chrom_sizes(chrom_sizes_fn):
    """
    Load a dictionary of chromosome lengths from a 2-column text file or a faidx file.

    Parameters
    ----------
    chrom_sizes_fn : str
        The file path to the chromosome size file. It should be a text file where each line
        contains a chromosome name and its size, separated by a tab.

    Returns
    -------
    dict
        A dictionary where the keys are chromosome names (str) and the values are chromosome sizes (int).
    """
    chrom_sizes = {}
    with open(chrom_sizes_fn) as f:
        for record in f:
            chrom, cs = record.strip().split('\t')[:2]
            chrom_sizes[chrom] = int(cs)
    return chrom_sizes
