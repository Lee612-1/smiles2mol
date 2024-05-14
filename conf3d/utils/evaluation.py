from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
import random


def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd


def filter_gen_list(gen_list, ref_list):
    filtered_list = []
    for gen_mol in gen_list:
        try:
            rmsd = GetBestRMSD(gen_mol, ref_list[0])
            filtered_list.append(gen_mol)
        except:
            pass
    return filtered_list


def get_cov_mat(gen_list, ref_list, threshold=0.5, useFF=False):
    if gen_list==[] or ref_list==[]:
        return None, None
    cov_count = 0
    mat_sum = 0
    for ref_mol in ref_list:
        rmsd_list = []
        for gen_mol in gen_list:
            if useFF == True:
                try:
                    MMFFOptimizeMolecule(gen_mol)
                except:
                    pass
            rmsd = GetBestRMSD(gen_mol, ref_mol)
            rmsd_list.append(rmsd)
        if min(rmsd_list)<=threshold:
            cov_count+=1
        mat_sum+=min(rmsd_list)
        
    return 100*cov_count/len(ref_list), mat_sum/len(ref_list)


def get_cov_mat_p(gen_list, ref_list, threshold=0.5, useFF=False):
    if gen_list==[] or ref_list==[]:
        return None, None
    cov_count = 0
    mat_sum = 0
    for gen_mol in gen_list:
        if useFF == True:
            try:
                MMFFOptimizeMolecule(gen_mol)
            except:
                pass
        rmsd_list = []
        for ref_mol in ref_list:        
            rmsd = GetBestRMSD(gen_mol, ref_mol)
            rmsd_list.append(rmsd)
        if min(rmsd_list)<=threshold:
            cov_count+=1
        mat_sum+=min(rmsd_list)
        
    return 100*cov_count/len(gen_list), mat_sum/len(gen_list)


def add_element(test_data):
    for i in range(len(test_data)):
        a = len(test_data[i][3])
        try:
            b = len(test_data[i][4])
            if b > 0 :
                for _ in range(2*a-b):
                    index = random.randint(0, len(my_list)-1)
                    value = test_data[i][4][index]
                    test_data[i][4].append(value)
        except:
            pass
    return None