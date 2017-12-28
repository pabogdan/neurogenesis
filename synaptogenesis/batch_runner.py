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

cases = [1, 2, 3]

parameters_of_interest = {'case':cases}

log_calls = []

for case in cases:
    for run in range(10):
        filename = "case{}_run{}" \
                   "_@{}".format(case,
                                 run+1,
                                 suffix)
        count += 1
        null = open(os.devnull, 'w')
        print "Run ", count, "..."
    
        call = [sys.executable,
            'topographic_map_formation.py',
            '--case', str(case),
            '-i', input_filename,
            '-o', filename,
            '--no_iterations',
            str(iterations),
            '--t_record',
            str(t_record)
            ]
        log_calls.append(call)
        subprocess.call(call,
                        stdout=null, stderr=null)
        print "Run", count, "complete."
print "All done!"

end_time = plt.datetime.datetime.now()
total_time = end_time - currrent_time
np.savez("batch_{}".format(suffix),
         parameters_of_interest=parameters_of_interest,
         total_time=total_time,
         log_calls=log_calls)
