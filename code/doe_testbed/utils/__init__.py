"""
doe_testbed.utils
====================
General-purpose utilities used internally by the testbed.

Only lightweight tools likely to be used directly (sampling, caching,
response-surface visualisation) are publicly exported.
"""

from doe_testbed.utils.cache import CacheManager
from doe_testbed.utils.sampling import lhs, sample, sobol_sequence
from doe_testbed.utils.visualization import plot_response_surface, save_figure

__all__ = [
    "sample",
    "lhs",
    "sobol_sequence",
    "CacheManager",
    "plot_response_surface",
    "save_figure",
]
