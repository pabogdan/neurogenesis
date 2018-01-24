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
t_record = 300000

cases = [1]
# input_types = [1, 3, 4]
# lesion_types = [1, 2]
sigma_stims = [.5,1., 1.5, 2., 2.5, 3.,3.5]
sigma_form_lats = [.5,1., 1.5, 2., 2.5, 3.,3.5]
no_runs = 1
lateral_inhibition = 1

parameters_of_interest = {
    'cases': cases,
    # 'input_types': input_types,
    # 'lesion_types': lesion_types,
    'no_runs':no_runs,
    # 'lateral_inhibitions':lateral_inhibition,
    'sigma_stims':sigma_stims,
    'sigma_form_lats':sigma_form_lats
}

log_calls = []

for case in cases:
    for sigma_stim in sigma_stims:
        for sigma_form_lat in sigma_form_lats:
            for run in range(no_runs):
                filename = "case{}_sstim{}_sformlat{}_run{}" \
                           "_@{}".format(case,
                                         sigma_stim,
                                         sigma_form_lat,
                                         run + 1,
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
                        str(t_record),
                        '--sigma_stim', str(sigma_stim),
                        '--sigma_form_lat', str(sigma_form_lat)
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
