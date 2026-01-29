class MultiHaplotypeValidator:
    """
    A class to validate and manage haplotypes, ensuring they belong to a predefined set of allowed haplotypes.
    It caches identical frozensets and returns references to the same object, to reduce memory usage

    Attributes:
    ----------
    allowed_haplotypes : frozenset
        A frozen set of allowed haplotypes.
    """

    def __init__(self, allowed_haplotypes=None):
        """
        Initializes the HaplotypeValidator with a set of allowed haplotypes.

        Parameters:
        -----------
        allowed_haplotypes : iterable or None
            A collection of allowed haplotypes (e.g., list, set) to be converted into a frozenset.
        """
        if allowed_haplotypes is not None:
            allowed_haplotypes = frozenset(allowed_haplotypes)
        self.allowed_haplotypes = allowed_haplotypes
        self._cached = {}

    def __getitem__(self, haps):
        """
        Retrieves a frozenset of haplotypes, ensuring they are a subset of allowed haplotypes.
        Caches identical frozensets and returns references to the same object, to reduce memory usage

        Parameters:
        -----------
        haps : iterable
            A collection of haplotypes (e.g., list, set) to be retrieved.

        Returns:
        --------
        frozenset
            A frozenset representing the validated haplotype set.
        """
        haps = frozenset(haps)
        if self.allowed_haplotypes is not None:
            haps = haps.intersection(self.allowed_haplotypes)
        if haps not in self._cached:
            self._cached[haps] = haps
        return self._cached[haps]

    def __eq__(self, other):
        return self.allowed_haplotypes == other

    def intersection(self, hap_a, hap_b):
        """
        Computes the intersection of two haplotype sets and validates the result.

        The intersection is first computed and then validated using the `__getitem__` method.

        Parameters:
        -----------
        hap_a : iterable
            A collection of haplotypes (e.g., list, set) for the first haplotype set.
        hap_b : iterable
            A collection of haplotypes (e.g., list, set) for the second haplotype set.

        Returns:
        --------
        frozenset
            A frozenset representing the validated intersection of `hap_a` and `hap_b`.

        Raises:
        -------
        KeyError
            If the intersection contains haplotypes not recognized as allowed.
        """
        hap_a, hap_b = frozenset(hap_a), frozenset(hap_b)
        return self[hap_a.intersection(hap_b)]

    def union(self, hap_a, hap_b):
        """
        Computes the union of two haplotype sets and validates the result.

        The union is first computed and then validated using the `__getitem__` method.

        Parameters:
        -----------
        hap_a : iterable
            A collection of haplotypes (e.g., list, set) for the first haplotype set.
        hap_b : iterable
            A collection of haplotypes (e.g., list, set) for the second haplotype set.

        Returns:
        --------
        frozenset
            A frozenset representing the validated union of `hap_a` and `hap_b`.

        Raises:
        -------
        KeyError
            If the union contains haplotypes not recognized as allowed.
        """
        hap_a, hap_b = frozenset(hap_a), frozenset(hap_b)
        return self[hap_a.union(hap_b)]
