import subprocess
import os
import numpy as np
import sys
import hashlib
import pylab as plt

currrent_time = plt.datetime.datetime.now()
string_time = currrent_time.strftime("%H%M%S_%d%m%Y")

suffix = hashlib.md5(string_time).hexdigest()

count = 0

input_filename = "../simulation_statistics/2009_09_04.17_48_33 32Syn300s/InitialConnectivity.mat"
iterations = 300000
# t_record = iterations
t_record = 30000

cases = [1, 2]

for case in cases:
    for b in [1.4, 1.3, 1.2, 1.1]:
        for t_minus in [30, 60, 90]:
            for f_peak in [152.8, 110, 60]:
                for sigma_stim in [2.5, 2, 1.5, 1]:
                    filename = "case{}_b{}_tminus{}_fpeak{}_sigmastim{}_@{}".format(case, b, t_minus, f_peak, sigma_stim, suffix)
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
                                     '--t_minus', str(t_minus),
                                     '--f_peak', str(f_peak),
                                     '--sigma_stim', str(sigma_stim)],
                                    stdout=null, stderr=null)
                    print "Run", count, "complete."
print "All done!"
