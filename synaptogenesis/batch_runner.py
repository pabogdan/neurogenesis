import subprocess
import os
import numpy as np
import sys

count = 0

input_filename = "../simulation_statistics/2009_09_04.17_48_33 32Syn300s/InitialConnectivity.mat"
iterations = 300000
# t_record = iterations
t_record = 30000

cases = [3]

for case in cases:
    for b in np.arange(1.4, 0.6, -0.05):
        for t_minus in np.arange(100, 0, -10):
            filename = "case{}_b{}_tminus{}".format(case, b, t_minus)
            count += 1
            null = open(os.devnull, 'w')
            print "Run ", count, "..."
            subprocess.call([sys.executable, 'topographic_map_formation.py',
                             '--case', str(case),
                             '-i', input_filename,
                             '-o', filename,
                             '--no_iterations', str(iterations),
                             '--t_record', str(t_record),
                             '--b', str(b),
                             '--t_minus', str(t_minus)],
                            stdout=null, stderr=null)
            print "Run", count, "complete."
print "All done!"
