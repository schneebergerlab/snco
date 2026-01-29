from collections import Counter


def weighted_chunks(items, weight_fn, processes, oversubscription=10):
    """
    Partition items into ~processes*oversubscription chunks of roughly equal total weight.

    Parameters
    ----------
    items : sequence
        Sequence of objects to split.
    weight_fn : callable
        Function mapping an item -> numeric weight.
    processes : int
        Number of worker processes.
    oversubscription : int, optional
        Oversubscription factor (default 10). Produces about processes*oversubscription chunks.

    Yields
    ------
    list
        A chunk (list of items).
    """
    weights = [weight_fn(x) for x in items]
    idx_sorted = sorted(range(len(items)), key=lambda i: weights[i], reverse=True)

    nchunks = max(processes * oversubscription, 1)
    chunks = [[] for _ in range(nchunks)]
    chunk_weights = [0] * nchunks

    for i in idx_sorted:
        j = min(range(nchunks), key=chunk_weights.__getitem__)
        chunks[j].append(items[i])
        chunk_weights[j] += weights[i]

    for c in chunks:
        if c:
            yield c


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