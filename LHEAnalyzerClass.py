
## -------------------------------------------------------------------------- ##
##    Author:    A.M.M Elsayed                                                ##
##    Email:     ahmedphysica@outlook.com                                     ##
##    Institute: University of Science and Technology of China                ##
## -------------------------------------------------------------------------- ##
##                                                                            ##
##    Updates On:                                                             ##
##    https://github.com/ammelsayed/LHEAnalyzer                               ##
## -------------------------------------------------------------------------- ##

import ROOT
from array import array
import pylhe
import os
import math

from tqdm import tqdm

class LHEAnalyzer:

    def __init__(self, lhe_file_path, max_events = None):

        print("Starting ..")
        self.lhe_file_path = lhe_file_path
        # self.total_n_events = pylhe.read_num_events(self.lhe_file_path)
        # self.max_num_events = max_events if max_events is not None else self.total_n_events
        # print("Total number of events:", self.max_num_events) 

        self.sm_leptons = [11, 13, 15, -11, -13, -15]
        self.sm_neutrinos = [12, 14, 16, -12, -14, -16]
        self.sm_quarks = list(range(1,7)) + [-x for x in range(1,7)]
        self.sm_truth_jets = [21] + self.sm_quarks # Gluon (21) + Quarks
        self.sm_ew_bosons = [22, 23, 24, -24, 25] # gamma, Z, W+, W-, H

        self.sm_objects = {
            "Electrons" : {"ID": [11, -11]},
            "Muons" : {"ID": [13, -13]},
            "Tauons" : {"ID": [15, -15]},
            "Leptons" : {"ID": self.sm_leptons},
            "Neutrinos" : {"ID": self.sm_neutrinos},

            "Top" : {"ID": [6, -6]},
            "Bottom" : {"ID": [5, -5]},
            "Charm" : {"ID": [4, -4]},
            "Strange" : {"ID": [3, -3]},
            "Down" : {"ID": [2, -2]},
            "Up" : {"ID": [1, -1]},
            "Gluons" : {"ID": [21]},
            "Quarks" : {"ID": self.sm_quarks}, 
            "Jets" : {"ID": self.sm_truth_jets},

            "Gamma" : {"ID": [22]},
            "Z" : {"ID": [23]},
            "Wp" : {"ID": [24]},
            "Wm" : {"ID": [-24]},
            "W" : {"ID": [24, -24]},
            "H" : {"ID": [25]}
        }

        self.bsm_objects = {
            "Sig0" : {"ID" : [9000018]},
            "Sigm" : {"ID" : [9000017]},
            "Sigp" : {"ID" : [-9000017]}
        }

        self.analysis_variable_names = [
            "E",
            "Mass",
            "Px",
            "Py",
            "Pz",
            "PT",
            "P",
            "Eta",
            "Phi"
        ]

        self.analyze_initial_states = True  # false by default
        self.analyze_intermediate_states = True # false by default
        self.analyze_final_states = True # true by default
    
    def include_initial_states(self, enable : bool) -> None:
        self.analyze_initial_states = enable
    
    def include_intermediate_states(self, enable : bool) -> None:
        self.analyze_intermediate_states = enable
    
    def include_final_states(self, enable : bool) -> None:
        self.analyze_final_states = enable

    def find_categories(self,pid):
        cats = []
        for name, info in self.sm_objects.items():
            if pid in info.get("ID", []):
                cats.append(name)
        # also include BSM matches if any
        for name, info in self.bsm_objects.items():
            if pid in info.get("ID", []):
                cats.append(name)

        return cats if cats else [f"PID_{pid}"]

    def get_status_name(self, status):
        return {-1: "Initial", 2: "Intermediate", 1: "Final" }.get(status)
     
    def get_particle_kinematics(self, p):

        # Extract components
        px = getattr(p, "px", 0.0)
        py = getattr(p, "py", 0.0)
        pz = getattr(p, "pz", 0.0)
        E  = getattr(p, "e", 0.0)
        m  = getattr(p, "m", 0.0)

        # compute derived quantities
        pt = math.sqrt(px*px + py*py)
        p_tot = math.sqrt(px*px + py*py + pz*pz)

        # pseudorapidity eta: careful about division by zero
        if p_tot != abs(pz):
            eta = 0.5 * math.log((p_tot + pz) / (p_tot - pz))
        else:
            eta = float('inf') if pz > 0 else float('-inf')

        phi = math.atan2(py, px)

        return {
            "E": [float(E)],
            "Mass": [float(m)],
            "Px": [float(px)],
            "Py": [float(py)],
            "Pz": [float(pz)],
            "PT": [float(pt)],
            "P": [float(p_tot)],
            "Eta": [float(eta)],
            "Phi": [float(phi)]
        }

    def get_root_file(self):

        # create file and directories
        root_file_path = "data.root"

        if not os.path.exists(root_file_path):

            output_root_file = ROOT.TFile("data.root", "RECREATE")
            output_root_file.mkdir("Initial")
            output_root_file.mkdir("Intermediate")
            output_root_file.mkdir("Final")

            # containers: directory_name -> { category_name: (tree, storage_dict) }
            trees = {"Initial": {}, "Intermediate": {}, "Final": {}}

            var_names = self.analysis_variable_names

          
            # helper: decide if we should add a Charge branch for this category
            def needs_charge_branch(category):
                return category in ("Electrons", "Muons", "Tauons", "Leptons")

            # Loop events
            for event_num, event in tqdm(enumerate(pylhe.read_lhe(self.lhe_file_path), start=1)):

                for p in event.particles:
                    pid = int(p.id)
                    status = int(p.status)
                    # find which directory based on status
                    dir_name = self.get_status_name(status)

                    # compute kinematics
                    res = self.get_particle_kinematics(p)

                    # Determine all categories this pid belongs to 
                    categories = self.find_categories(pid)
                    
                    for category in categories:
                        tree_name = category.replace(" ", "_")
                        if category not in trees[dir_name]:
                            tree = ROOT.TTree(tree_name, f"{category} ({dir_name})")
                            stor = {}

                            # float branches for kinematic variables
                            for name in var_names:
                                stor[name] = array('f', [0.0])
                                tree.Branch(name, stor[name], f"{name}/F")

                            # integer branches
                            stor["Status"] = array('i', [0])
                            stor["PID"] = array('i', [0])
                            stor["EventNumber"] = array('i', [0])
                            tree.Branch("Status", stor["Status"], "Status/I")
                            tree.Branch("PID", stor["PID"], "PID/I")
                            tree.Branch("EventNumber", stor["EventNumber"], "EventNumber/I")

                            # add Charge branch for leptons (collective or specific)
                            if category in ["Electrons", "Muons", "Tauons", "Leptons"]:
                                stor["Charge"] = array('i', [0])
                                tree.Branch("Charge", stor["Charge"], "Charge/I")

                            trees[dir_name][category] = (tree, stor)

                        # fill the tree
                        tree, stor = trees[dir_name][category]
                        for name in var_names:
                            stor[name][0] = float(res.get(name, [0.0])[0])

                        stor["Status"][0] = status
                        stor["PID"][0] = pid
                        stor["EventNumber"][0] = event_num

                        # charge: PDG sign convention -> for e/mu/tau: positive PDG id corresponds to negatively charged particle
                        if "Charge" in stor:
                            # set -1 for particle (pid > 0) e-/mu-/tau- ; +1 for anti-particle (pid < 0)
                            charge = -1 if pid > 0 else 1
                            stor["Charge"][0] = int(charge)

                        tree.Fill()

            # Write trees into their directories and close file
            for dir_name in ("Initial", "Intermediate", "Final"):
                output_root_file.cd(dir_name)    # enter the directory
                for category, (tree, _) in trees[dir_name].items():
                    tree.Write()

            output_root_file.Close()
