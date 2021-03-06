# AUTOGENERATED! DO NOT EDIT! File to edit: 65_AFT_error_distributions.ipynb (unless otherwise specified).

__all__ = ['get_distribution']

# Cell
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell
def get_distribution(dist:str):
    """
    Get the logpdf and logcdf of a given torch distribution
    """
    dist = getattr(torch.distributions, dist.title())
    if not isinstance(dist.support, torch.distributions.constraints._Real):
        raise Exception("Distribution needs support over ALL real values.")

    dist = partial(dist, loc=0.0)

    def dist_logpdf(ξ, σ):
        return dist(scale=σ).log_prob(ξ)

    def dist_logicdf(ξ, σ):
        """
        log of inverse cumulative distribution function
        """
        return torch.log(1 - dist(scale=σ).cdf(ξ))

    return dist_logpdf, dist_logicdf