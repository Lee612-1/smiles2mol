from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from conf3d import dataset


######  要修改
def validate(text, smiles):
    lines = text.split('\n')
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    num_atoms = int(lines[0])
    if lines[1] != '' or lines[num_atoms+1] == ' ' or lines[num_atoms+2] !='':
        return False
    
    atoms = []
    for line in lines[2:num_atoms+2]:
        try:
            atom, x, y, z = line.split()
            atoms.append(atom)
            x, y, z = float(x), float(y), float(z)
        except:
            return False

    rd_atoms = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_symbol = atom.GetSymbol()
        rd_atoms.append(atom_symbol)
    
    if set(atoms) != set(rd_atoms):
        return False

    return True


def calc_rmsd(generated_text, text, smiles):
    with open('ref.xyz', 'w') as file:
            file.write(text)
    with open('gen.xyz', 'w') as file:
            file.write(generated_text)
    try:
        raw_ref_mol=Chem.MolFromXYZFile('ref.xyz')
        raw_gen_mol=Chem.MolFromXYZFile('gen.xyz')
        ref_mol = Chem.Mol(raw_ref_mol)
        gen_mol = Chem.Mol(raw_gen_mol)
    except:
        return None
    try:
        rdDetermineBonds.DetermineBonds(gen_mol)
    except Exception as e:
        # print('generated molecule is invalid:', e)
        return None
    MMFFOptimizeMolecule(gen_mol)
    try:
        rdDetermineBonds.DetermineBonds(ref_mol)
    except Exception as e:
        # print('refer molecule is invalid:', e)
        return None 
    gen_mol = RemoveHs(gen_mol)
    ref_mol = RemoveHs(ref_mol) 
    try:
        rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    except Exception as e:
        # print('generated molecule is invalid:', e)
        return None
    # if Chem.MolToSmiles(ref_mol) != smiles:
    #     return None
    return rmsd


def get_atom_order(text):  
    lines = text.split('\n')
    num_atoms = int(lines[0])
    atoms = []
    try:
        for line in lines[2:num_atoms+2]:
            atom = line.split()[0]
            atoms.append(atom)
        return atoms
    except:
        return []