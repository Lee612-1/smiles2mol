from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule


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


def get_cov_mat(gen_list, ref_list, threshold=0.5):
    if gen_list==[] or ref_list==[]:
        return None, None
    cov_count = 0
    mat_sum = 0
    for ref_mol in ref_list:
        rmsd_list = []
        for gen_mol in gen_list:
            rmsd = GetBestRMSD(gen_mol, ref_mol)
            rmsd_list.append(rmsd)
        if min(rmsd_list)<=threshold:
            cov_count+=1
        mat_sum+=min(rmsd_list)
        
    return 100*cov_count/len(ref_list), mat_sum/len(ref_list)

def get_cov_mat_p(gen_list, ref_list, threshold=0.5):
    if gen_list==[] or ref_list==[]:
        return None, None
    cov_count = 0
    mat_sum = 0
    for gen_mol in gen_list:
        rmsd_list = []
        for ref_mol in ref_list:        
            rmsd = GetBestRMSD(gen_mol, ref_mol)
            rmsd_list.append(rmsd)
        if min(rmsd_list)<=threshold:
            cov_count+=1
        mat_sum+=min(rmsd_list)
        
    return 100*cov_count/len(gen_list), mat_sum/len(gen_list)
    