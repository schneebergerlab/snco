from collections import Counter


def genotyping_results_formatter(genotypes):
    """
    Format the genotyping results into a human-readable string.
    """
    geno_counts = Counter(genotypes.values())
    fmt = 'Genotyping results:\n'
    ljust_size = max([len(g) for g in geno_counts.keys()]) + 5
    ljust_size = max(ljust_size, 12)
    fmt += f'   Genotype'.ljust(ljust_size)
    fmt += 'Num. barcodes\n'
    for geno, count in geno_counts.most_common():
        fmt += f'   {geno}'.ljust(ljust_size)
        fmt += f'{count}\n'
    return fmt