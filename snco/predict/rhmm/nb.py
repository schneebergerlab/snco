import torch
from pomegranate._utils import (
    _cast_as_tensor, _cast_as_parameter, _update_parameter,
    _check_parameter, _reshape_weights
)
from pomegranate.distributions._distribution import Distribution


class NegativeBinomial(Distribution):
    """
    A DESeq2-style Negative Binomial distribution parameterized by mean and dispersion.
    Variance is given by:
        Var(X) = mean + alpha * mean^2
    Parameters
    ----------
    means : list, numpy.ndarray, torch.Tensor or None, shape=(d,)
        Mean count for each feature.
    alphas : list, numpy.ndarray, torch.Tensor or None, shape=(d,)
        Dispersion factor for each feature.
    inertia : float
        Interpolation factor for updates (0 = replace, 1 = ignore update).
    frozen : bool
        If True, disables training.
    check_data : bool
        If True, validates input types and values.
    """

    def __init__(self, means=None, alphas=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "NegativeBinomial"

        self.means = _check_parameter(_cast_as_parameter(means), "means", min_value=0, ndim=1)
        self.alphas = _check_parameter(_cast_as_parameter(alphas), "alphas", min_value=0, ndim=1)

        self._initialized = self.means is not None and self.alphas is not None
        self.d = self.means.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        self.means = _cast_as_parameter(torch.ones(d, dtype=self.dtype, device=self.device))
        self.alphas = _cast_as_parameter(torch.ones(d, dtype=self.dtype, device=self.device) * 0.1)
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        if not self._initialized:
            return

        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_x2w_sum', torch.zeros(self.d, device=self.device))

    def sample(self, n):
        """
        Sample n values using mean and dispersion.
        Internally maps to (r, p) form:
            r = 1 / alpha
            p = r / (r + mean)
        """
        r = 1.0 / self.alphas
        p = r / (r + self.means)
        dist = torch.distributions.NegativeBinomial(total_count=r, probs=(1 - p))
        return dist.sample((n,))

    def log_probability(self, X):
        """
        Compute log-probabilities for each example under the NB distribution.
        """
        X = _check_parameter(
            _cast_as_tensor(X), "X", min_value=0.0, ndim=2, shape=(-1, self.d),
            check_parameter=self.check_data
        )

        r = 1.0 / self.alphas
        p = r / (r + self.means)

        return torch.sum(
            torch.lgamma(X + r) - torch.lgamma(r) - torch.lgamma(X + 1) +
            r * torch.log(p) + X * torch.log1p(-p),
            dim=-1
        )

    def summarize(self, X, sample_weight=None):
        """
        Accumulate sufficient statistics (x, x^2, weights).
        """
        if self.frozen:
            return

        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        _check_parameter(X, "X", min_value=0, check_parameter=self.check_data)

        self._w_sum += torch.sum(sample_weight, dim=0)
        self._xw_sum += torch.sum(X * sample_weight, dim=0)
        self._x2w_sum += torch.sum((X ** 2) * sample_weight, dim=0)

    def from_summaries(self):
        """
        Update mean and alpha from accumulated statistics using method-of-moments
        """
        if self.frozen:
            return

        mean = self._xw_sum / self._w_sum
        second_moment = self._x2w_sum / self._w_sum
        var = second_moment - mean ** 2

        # Ensure variance is always > mean for NB validity
        var = torch.clamp(var, min=mean + 1e-6)
        alpha = (var - mean) / (mean ** 2)
        alpha = torch.clamp(alpha, min=1e-8)

        _update_parameter(self.means, mean, self.inertia)
        _update_parameter(self.alphas, alpha, self.inertia)

        self._reset_cache()