def edit_dist(umi1, umi2):
    """
    Calculate the edit (Hamming) distance between two unique molecular identifiers (UMIs).

    Parameters
    ----------
    umi1 : str
        First UMI string (DNA sequence).
    umi2 : str
        Second UMI string (DNA sequence).

    Returns
    -------
    int
        Number of differing positions between the two UMIs.
    """
    return sum(i != j for i, j in zip(umi1, umi2))

