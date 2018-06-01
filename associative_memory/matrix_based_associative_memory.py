from __future__ import division, print_function
import numpy as np

from associative_memory.pattern_generator import PatternGenerator
from associative_memory.memory_analyzer import MemoryAnalyzer


class DenselyConnectedAssociativeMatrix(object):
    def __init__(self, m_in, n_in, m_out, n_out, threshold, new_neurons=False,
                 turnover=False, noisy_readout=None, seed=None):
        self.random = np.random.RandomState(seed=seed)
        self.m_in = m_in
        self.n_in = n_in
        self.m_out = m_out
        self.n_out = n_out
        self.threshold = threshold
        self.new_neurons = new_neurons
        self.turnover = turnover
        self.noisy_readout = noisy_readout
        self.memory = np.zeros((m_in, m_out))
        self.pattern_generator = PatternGenerator(self.random.randint(2 ** 31))

    def store_pattern(self, input_pattern, desired_output_pattern=None):
        # store the input Nin-of-Min pattern in the memory matrix with the
        # possibility to
        if desired_output_pattern:
            clean_output_pattern = desired_output_pattern
        else:
            clean_output_pattern = \
                self.pattern_generator.generate_pattern(n=self.n_out,
                                                        m=self.m_out)
        memory_locations = np.matmul(
            input_pattern.astype(bool).reshape(input_pattern.size, 1),
            clean_output_pattern.astype(bool).reshape(
                1, clean_output_pattern.size))
        self.memory[memory_locations
            ] = 1
        return clean_output_pattern

    def associated_pattern_threshold(self, input_pattern):
        # input the Nin-of-Min pattern and see what pattern is reported back
        # decision is based on the thresholded neural responses
        non_thresholded_vector = \
            np.sum(self.memory[input_pattern.astype(bool), :], axis=0)
        if self.noisy_readout:
            non_thresholded_vector += self.random.randint(
                -self.noisy_readout, self.noisy_readout + 1,
                size=non_thresholded_vector.size)
        threshold_vector = non_thresholded_vector >= self.threshold
        return threshold_vector.astype(np.double)

    def associated_patern_max_activation(self, input_pattern):
        # input the Nin-of-Min pattern and see what pattern is reported back
        # decision is based on the maximum output neural responses
        # after thresholding
        non_thresholded_vector = \
            np.sum(self.memory[input_pattern.astype(bool), :], axis=0)
        if self.noisy_readout:
            non_thresholded_vector += self.random.randint(
                -self.noisy_readout, self.noisy_readout + 1,
                size=non_thresholded_vector.size)
        thresholded_vector = np.argwhere(non_thresholded_vector >= \
                                         self.threshold)
        ordered_responses = np.argsort(non_thresholded_vector)[::-1]
        selected_reponse = np.zeros(self.m_out)
        selection_mask = np.intersect1d(ordered_responses[:self.n_out],
                                        thresholded_vector)
        selected_reponse[selection_mask] = 1
        return selected_reponse


if __name__ == "__main__":
    n_in = 10
    m_in = 100
    n_out = 1
    m_out = 10
    memory_unit = DenselyConnectedAssociativeMatrix(m_in, n_in, m_out,
                                                    n_out, n_in,
                                                    noisy_readout=2,
                                                    seed=9999)
    single_pattern = np.concatenate((np.ones(n_in), np.zeros(m_in - n_in)))
    output_pattern = memory_unit.store_pattern(single_pattern)
    op_1 = memory_unit.associated_patern_max_activation(single_pattern)
    single_pattern2 = np.random.permutation(single_pattern)
    output_pattern2 = memory_unit.store_pattern(single_pattern2)
    op_2 = memory_unit.associated_patern_max_activation(single_pattern2)

    print("{:20}".format("output_pattern:"), output_pattern)
    print("{:20}".format("op1"), op_1)
    # assert (np.all(output_pattern == op_1))
    print("{:20}".format("output_pattern2:"), output_pattern2)
    print("{:20}".format("op2"), op_2)
    # assert (np.all(output_pattern2 == op_2))
    print("memory content:\n", memory_unit.memory)

    memory_analyzer = MemoryAnalyzer()
    s = memory_analyzer.analyze_saturatation(memory_unit, 2)
    print("saturation", s)

    n_in = 20
    m_in = 1000
    n_out = 10
    m_out = 500
    seed = 9999
    in_patterns = []
    out_patterns = []
    immediate_out_patterns = []
    memory_unit = DenselyConnectedAssociativeMatrix(m_in, n_in, m_out,
                                                    n_out, threshold=n_in - 5,
                                                    noisy_readout=2,
                                                    seed=seed)
    p_gen = PatternGenerator(seed=seed)
    temp_pattern = p_gen.generate_pattern(n=n_in, m=m_in)
    in_patterns.append(temp_pattern)
    out_patterns.append(memory_unit.store_pattern(temp_pattern))