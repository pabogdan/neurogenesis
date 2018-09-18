"""
Runtime analysis for testing and training of movement detection networks

Author: Petrut A. Bogdan

"""
# Imports
from zipfile import BadZipfile

from common_analysis_imports import *
from os import listdir
from os.path import isfile, join
start_time = plt.datetime.datetime.now()

save_dir = "runtime_analysis_meta"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
synaptogenesis_folder_relative_location = "../synaptogenesis/"
onlyfiles = [f for f in listdir(synaptogenesis_folder_relative_location) if
             isfile(join(synaptogenesis_folder_relative_location, f))]

only_delay_npz_files = [f for f in onlyfiles if ".npz" in f and "delay" in f]

testing_random_delay_files = []
testing_constant_delay_files = []
training_random_delay_files = []
training_constant_delay_files = []
exceptions = []
ke_count = 0
io_count = 0
bzf_count = 0
exception_strings = []
for f in only_delay_npz_files:
    try:
        is_constant_delay = False
        pathed_file = join(synaptogenesis_folder_relative_location, f)
        data = np.load(pathed_file)
        if (data['exception'] and
                not data['exception'].ravel()[0] == "None") \
                or "error_" in f:
            # Broken
            exceptions.append((f, data['exception']))
        elif data['testing']:
            # Testing
            sim_params = np.array(data['sim_params']).ravel()[0]
            if 'delay_interval' in sim_params.keys():
                is_constant_delay = sim_params['delay_interval'] == [1, 1]
            else:
                is_constant_delay = "constant" in f
            if is_constant_delay:
                testing_constant_delay_files.append((
                    f, sim_params['simtime'],
                    np.array(data['total_time']).ravel()[
                        0].total_seconds() * 1000,
                    sim_params,
                    os.stat(pathed_file)))
            else:
                testing_random_delay_files.append((
                    f, sim_params['simtime'],
                    np.array(data['total_time']).ravel()[
                        0].total_seconds() * 1000,
                    sim_params,
                    os.stat(pathed_file)))
        elif not data['testing']:
            # Training
            sim_params = np.array(data['sim_params']).ravel()[0]
            if 'delay_interval' in sim_params.keys():
                is_constant_delay = sim_params['delay_interval'] == [1, 1]
            else:
                is_constant_delay = "constant" in f
            if is_constant_delay:
                training_constant_delay_files.append((
                    f, sim_params['simtime'],
                    np.array(data['total_time']).ravel()[
                        0].total_seconds() * 1000,
                    sim_params,
                    os.stat(pathed_file)))
            else:
                training_random_delay_files.append((
                    f, sim_params['simtime'],
                    np.array(data['total_time']).ravel()[
                        0].total_seconds() * 1000,
                    sim_params,
                    os.stat(pathed_file)))
        else:
            sim_params = np.array(data['sim_params']).ravel()[0]
            print("Something went wrong with", f, sim_params)
            break
    except KeyError as ke:
        ke_count += 1
    except IOError as io:
        io_count += 1
    except BadZipfile as bzf:
        bzf_count += 1
    except Exception as e:
        exception_strings.append(str(e))
        traceback.print_exc()

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
suffix = end_time.strftime("_%H%M%S_%d%m%Y")
if args.filename:
    filename = args.filename
else:
    filename = join(save_dir, "runtime_analysis_drifting_grating" + str(
        suffix))
np.savez(filename,
         testing_random_delay_files = testing_random_delay_files,
         testing_constant_delay_files=testing_constant_delay_files,
         training_random_delay_files=training_random_delay_files,
         training_constant_delay_files=training_constant_delay_files,
         exceptions=exceptions,
         keyerror_count=ke_count,
         ioerror_count=io_count,
         badzipfileerror_count=bzf_count,
         exception_strings=exception_strings)

print("Results in", filename)
print("Total time elapsed -- " + str(total_time))



