import os

from pathlib import Path
import shutil


root_folder = "/home/arnaud/Documents/ACS/Adaptive-Cutsel-MILP-main/ACS_instances"
instances_folder = "/home/arnaud/Documents/ACS/Adaptive-Cutsel-MILP-main/Instances/Instances"
solutions_folder = "/home/arnaud/Documents/ACS/Adaptive-Cutsel-MILP-main/Instances/Solutions"


ACS_list = list(Path(root_folder).rglob("*.gz" ))

for path in ACS_list:
    name = str(path).split("ACS_instances")[1]
    name_sol = name.split(".mps")[0] + ".sol.gz"

    instance_path = instances_folder + name
    sol_path = solutions_folder + name_sol

    target_instance_path = "/home/arnaud/Documents/ACS/Adaptive-Cutsel-MILP-main/Instances_ACS/Instances" + name
    target_solution_path = "/home/arnaud/Documents/ACS/Adaptive-Cutsel-MILP-main/Instances_ACS/Solutions"+ name_sol

    shutil.copyfile(instance_path, target_instance_path)
    shutil.copyfile(sol_path, target_solution_path)
   
