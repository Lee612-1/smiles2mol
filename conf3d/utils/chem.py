from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from torchvision.transforms.functional import to_tensor
import copy
import torch
import rdkit.Chem.Draw
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol, GetPeriodicTable
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from typing import List, Tuple

def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd

def draw_mol_image(rdkit_mol, tensor=False):
    rdkit_mol.UpdatePropertyCache()
    img = rdkit.Chem.Draw.MolToImage(rdkit_mol, kekulize=False)
    if tensor:
        return to_tensor(img)
    else:
        return img

def draw_mol_svg(mol,molSize=(450,150),kekulize=False):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        DP.Compute2DCoords(mc)
    drawer = MD2.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    # return svg.replace('svg:','')
    return svg