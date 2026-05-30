"""ResMill: Rule-based 3D geological reservoir modeling."""

__version__ = "0.1.2"

from .layers import (Layer, LobeLayer, GaussianLayer,
                     ChannelLayer,
                     DeltaLayer, DELTA_FAN)
from .reservoir import Reservoir
from .plotting import (plot_cube_slices, plot_slices, plot_layer, plot_reservoir,
                       alluvsim_cmap, alluvsim_legend_handles,
                       ALLUVSIM_FACIES_NAMES, ALLUVSIM_FACIES_COLORS,
                       RESMILL_CMAP)
