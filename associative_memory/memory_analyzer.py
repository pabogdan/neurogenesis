from __future__ import division
import numpy as np


class MemoryAnalyzer(object):
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)

    def analyze_saturatation(self, memory, number_of_patterns):
        p_in = memory.n_in / memory.m_in
        p_out = memory.n_out / memory.m_out
        saturation = 1. - (1.-p_in*p_out)**number_of_patterns
        return saturation


# if __name__ == "__main__":
