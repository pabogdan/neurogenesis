import subprocess
import os
import numpy as np
import sys
import hashlib
import pylab as plt
from batch_argparser import *

# Current defaults for [App: Motion detection]
# as of 26.09.2018

currrent_time = plt.datetime.datetime.now()
string_time = currrent_time.strftime("%H%M%S_%d%m%Y")

if args.suffix:
    suffix = args.suffix
else:
    suffix = hashlib.md5(string_time).hexdigest()

# Some constants
NO_CPUS = args.no_cpus
MAX_CONCURRENT_PROCESSES = args.max_processes

TRAINING_PHASE = 0
TESTING_PHASE = 1
PHASES = [TRAINING_PHASE, TESTING_PHASE]
PHASES_NAMES = ["training", "testing"]
PHASES_ARGS = ["--output", "--testing"]
# Constant for current batch purpose. Should be changed accordingly
S_MAX = 128
G_MAX = 0.1

concurrently_active_processes = 0

iterations = args.no_iterations  # trying to optimise fastest run

sigma_form_ffs = np.arange(2, 10.5, .5)
sigma_form_lats = np.copy(sigma_form_ffs)

# Compute total number of runs
total_runs = sigma_form_ffs.size * sigma_form_lats.size * len(PHASES)

parameters_of_interest = {
    'sigma_form_ff': sigma_form_ffs,
    'sigma_form_lat': sigma_form_lats
}

log_calls = []

for phase in PHASES:
    for sigma_form_ff in sigma_form_ffs:
        for sigma_form_lat in sigma_form_lats:
            filename = "random_delay_" \
                "smax_{}_" \
                "gmax_{}_" \
                "sigmaformff_{}_" \
                "sigmaformlat_{}_" \
                "_@{}".format(S_MAX,
                              G_MAX,
                              sigma_form_ff,
                              sigma_form_lat,
                              suffix)
            concurrently_active_processes += 1
            null = open(os.devnull, 'w')
            print("Run ", concurrently_active_processes, "...")

            call = [sys.executable,
                    'random_delays_drifting_gratings.py',
                    PHASES_ARGS[phase], filename,
                    '--s_max', str(S_MAX),
                    '--g_max', str(G_MAX),
                    '--no_iterations', str(iterations),
                    '--sigma_form_ff', str(sigma_form_ff),
                    '--sigma_form_lat', str(sigma_form_lat)
                    ]
            log_calls.append(call)
            if concurrently_active_processes % MAX_CONCURRENT_PROCESSES == 0\
                    or concurrently_active_processes == total_runs:
                # Blocking
                subprocess.call(call,
                                stdout=null, stderr=null)
                print("{} sims done".format(concurrently_active_processes))
            else:
                # Non-blocking
                subprocess.Popen(call,
                                 stdout=null, stderr=null)
print("All done!")

end_time = plt.datetime.datetime.now()
total_time = end_time - currrent_time
np.savez_compressed("batch_{}".format(suffix),
         parameters_of_interest=parameters_of_interest,
         total_time=total_time,
         log_calls=log_calls)
