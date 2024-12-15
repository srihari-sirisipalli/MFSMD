# import os
# import winreg as reg

# # Path to remove
# path_to_remove = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\Machine Fault Detection and Monitoring System"

# # Remove from the current environment
# pythonpath = os.environ.get("PYTHONPATH", "")
# paths = pythonpath.split(os.pathsep)
# if path_to_remove in paths:
#     paths.remove(path_to_remove)
#     new_pythonpath = os.pathsep.join(paths)
#     os.environ["PYTHONPATH"] = new_pythonpath
#     print(f"Removed '{path_to_remove}' from current session PYTHONPATH.")
# else:
#     print(f"'{path_to_remove}' not found in the current session PYTHONPATH.")

# # Update permanently in Windows Environment Variables
# key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Environment", 0, reg.KEY_SET_VALUE)
# try:
#     reg.SetValueEx(key, "PYTHONPATH", 0, reg.REG_SZ, new_pythonpath)
#     print(f"Removed '{path_to_remove}' from permanent PYTHONPATH.")
# finally:
#     reg.CloseKey()

import os

# Define the path to your virtual environment
venv_path = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\venv"

# Path to pyvenv.cfg
pyvenv_cfg = os.path.join(venv_path, "pyvenv.cfg")

# Read and modify pyvenv.cfg if it contains the path
if os.path.exists(pyvenv_cfg):
    with open(pyvenv_cfg, "r") as file:
        lines = file.readlines()
    with open(pyvenv_cfg, "w") as file:
        for line in lines:
            # Remove or modify any line referencing the unwanted path
            if "Machine Fault Detection and Monitoring System" not in line:
                file.write(line)

print(f"Updated {pyvenv_cfg} to remove unwanted paths.")


from src.feature_extraction.time_domain import TimeDomainFeatures
import inspect

# Print the source of the loaded class
print(inspect.getsource(TimeDomainFeatures))
import sys

print("Python Module Search Paths:")
for path in sys.path:
    print(path)
