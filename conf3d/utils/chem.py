from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
# from torchvision.transforms.functional import to_tensor
# import copy
# import torch
# import rdkit.Chem.Draw
# from rdkit.Chem.rdchem import Mol, GetPeriodicTable
# from rdkit.Chem.Draw import rdMolDraw2D as MD2


def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd

def get_atom(mol):
    atom_num = mol.GetNumAtoms() 
    bond_num = mol.GetNumBonds()
    mol_block = Chem.MolToMolBlock(mol).split('\n')
    order_list = mol_block[:4]
    for i in range(atom_num):
        order_list.append(mol_block[i+4][31:])
    return order_list

def get_bond(mol):
    atom_num = mol.GetNumAtoms() 
    bond_num = mol.GetNumBonds()
    mol_block = Chem.MolToMolBlock(mol).split('\n')
    order_list=mol_block[4+atom_num:]
    return order_list

def get_diff_bond(mol_list):
    bond_list = [list(t) for t in set(tuple(get_bond(mol)) for mol in mol_list)]
    return bond_list

    
# def draw_mol_image(rdkit_mol, tensor=False):
#     rdkit_mol.UpdatePropertyCache()
#     img = rdkit.Chem.Draw.MolToImage(rdkit_mol, kekulize=False)
#     if tensor:
#         return to_tensor(img)
#     else:
#         return img

# def draw_mol_svg(mol,molSize=(450,150),kekulize=False):
#     mc = Chem.Mol(mol.ToBinary())
#     if kekulize:
#         try:
#             Chem.Kekulize(mc)
#         except:
#             mc = Chem.Mol(mol.ToBinary())
#     if not mc.GetNumConformers():
#         DP.Compute2DCoords(mc)
#     drawer = MD2.MolDraw2DSVG(molSize[0],molSize[1])
#     drawer.DrawMolecule(mc)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()
#     # It seems that the svg renderer used doesn't quite hit the spec.
#     # Here are some fixes to make it work in the notebook, although I think
#     # the underlying issue needs to be resolved at the generation step
#     # return svg.replace('svg:','')
#     return svg