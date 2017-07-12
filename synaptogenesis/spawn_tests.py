try:
    import subprocess32 as subprocess
except:
    import subprocess

subprocess.Popen(["python", "topographic_map_formation.py", "--no_plot", "--case", "1"])
subprocess.Popen(["python", "topographic_map_formation.py", "--no_plot", "--case", "2"])
subprocess.Popen(["python", "topographic_map_formation.py", "--no_plot", "--case", "3"])
subprocess.Popen(["python", "topographic_map_formation.py", "--no_plot", "--case", "1", "--tau_refract", "2.0"])
