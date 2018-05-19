import numpy as np


class TileCoding:
    def __init__(self, n_tiling=4, range_width_tiling=[0, 1], n_width_tiling=4):
        self.n_tiling = n_tiling
        self.range_width_tiling = range_width_tiling
        self.n_width_tiling = n_width_tiling
