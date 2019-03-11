import numpy as np


class PatternGenerator(object):
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)

    def generate_pattern(self, n, m):
        return self.random.permutation(np.concatenate(
            (np.ones(n), np.zeros(m - n))))


if __name__ == "__main__":
    pg1 = PatternGenerator(seed=42)
    pg2 = PatternGenerator(seed=42)
    assert (
        np.all(pg1.generate_pattern(10, 100) == pg2.generate_pattern(10, 100)))
