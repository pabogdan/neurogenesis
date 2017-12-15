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

parameters_of_interest = ['case', 'b', 'gaussian_input', 'insult',
                          'random_partner']

log_calls = []

for case in cases:
    for b in [1.4, 1.3, 1.2, 1.1, 1.0]:
        for gaussian_input in [0, 1]:
            for insult in [0, 1]:
                for random_partner in [0, 1]:
                    for p_elim_dep in [0.0145, 0.0245, 0.0345, 0.0445]:
                        filename = "case{}_b{}" \
                                   "_gaussianinput{}" \
                                   "_insult" \
                                   "_randompartner{}" \
                                   "_pelimdep{}" \
                                   "_@{}".format(case,
                                                 b,
                                                 gaussian_input,
                                                 insult,
                                                 random_partner,
                                                 p_elim_dep,
                                                 suffix)
                        count += 1
                        null = open(os.devnull, 'w')
                        print "Run ", count, "..."

                        flags = "{} {} {}".format(
                            "--gaussian_input" if gaussian_input else "",
                            "--random_partner" if random_partner else "",
                            "--insult" if insult else "").strip(' ')
                        if flags == "":
                            call = [sys.executable,
                                    'topographic_map_formation.py',
                                    '--case', str(case),
                                    '-i', input_filename,
                                    '-o', filename,
                                    '--no_iterations',
                                    str(iterations),
                                    '--t_record', str(t_record),
                                    '--b', str(b),
                                    '--p_elim_dep', str(p_elim_dep)
                                    ]
                            log_calls.append(call)
                            subprocess.call(call,
                                            stdout=null, stderr=null)
                        else:
                            call = [sys.executable,
                                    'topographic_map_formation.py',
                                    '--case', str(case),
                                    '-i', input_filename,
                                    '-o', filename,
                                    '--no_iterations',
                                    str(iterations),
                                    '--t_record', str(t_record),
                                    '--b', str(b),
                                    '--p_elim_dep', str(p_elim_dep),
                                    flags
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
