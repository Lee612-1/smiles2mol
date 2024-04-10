import numpy as np
import random
import os
import copy
import json
from tqdm import tqdm
import pickle
from rdkit import Chem
import pandas as pd
from collections import defaultdict

def preprocess_GEOM_dataset(base_path, dataset_name, conf_per_mol=5, train_size=0.8, tot_mol_size=50000, seed=None):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name in [qm9, drugs]
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    tot_mol_size: max num of mols. The total number of final confs should be tot_mol_size * conf_per_mol
    seed: rand seed for RNG
    """

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < conf_per_mol:
            continue
        num_mols += 1
        num_confs += conf_per_mol
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)

    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs, use %d molecules with %d confs' % (num_mols, num_confs, tot_mol_size, tot_mol_size*conf_per_mol))


    # 1. select the most probable 'conf_per_mol' confs of each 2D molecule
    # 2. split the dataset based on 2D structure, i.e., test on unseen graphs
    train_data, val_data, test_data = [], [], []
    val_size = test_size = (1. - train_size) / 2

    # generate train, val, test split indexes
    split_indexes = list(range(tot_mol_size))
    random.shuffle(split_indexes)
    index2split = {}
    for i in range(0, int(len(split_indexes) * train_size)):
        index2split[split_indexes[i]] = 'train'
    for i in range(int(len(split_indexes) * train_size), int(len(split_indexes) * (train_size + val_size))):
        index2split[split_indexes[i]] = 'val'
    for i in range(int(len(split_indexes) * (train_size + val_size)), len(split_indexes)):
        index2split[split_indexes[i]] = 'test'        


    num_mols = np.zeros(4, dtype=int) # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int) # (tot, train, val, test)


    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        if mol.get('uniqueconfs') == conf_per_mol:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'conf_per_mol' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            rd_mol = conf_meta.get('rd_mol')
            data = [smiles, Chem.MolToSmiles(rd_mol), rd_mol.GetNumAtoms(), rd_mol.GetNumBonds(), Chem.MolToMolBlock(rd_mol)]
            datas.append(data)
        assert len(datas) == conf_per_mol

        if index2split[i] == 'train':
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif index2split[i] == 'val':    
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        elif index2split[i] == 'test': 
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)] 
        else:
            raise ValueError('unknown index2split value.')                         

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))    
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))    
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))    
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))    
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data, index2split



def get_GEOM_testset(base_path, dataset_name, block, tot_mol_size=200, seed=None, confmin=50, confmax=500):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name, should be in [qm9, drugs]
    block: block the training and validation set
    tot_mol_size: size of the test set
    seed: rand seed for RNG
    confmin and confmax: range of the number of conformations
    """

    #block smiles in train / val 
    block_smiles = defaultdict(int)
    for block_ in block:
        for i in range(len(block_)):
            block_smiles[block_[i][0]] = 1

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < confmin or u_conf > confmax:
            continue
        if block_smiles[smiles] == 1:
            continue

        num_mols += 1
        num_confs += u_conf
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)


    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs' % (num_mols, num_confs))


    bad_case = 0
    all_test_data = []
    num_valid_mol = 0
    num_valid_conf = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')
        no_canonicalize_smiles = True
        conf_ids = np.arange(mol.get('uniqueconfs'))
        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            rd_mol = conf_meta.get('rd_mol')
            if no_canonicalize_smiles:
                canonicalize_smiles = Chem.MolToSmiles(rd_mol)
                num_atom = rd_mol.GetNumAtoms()
                num_bond = rd_mol.GetNumBonds()
                no_canonicalize_smiles = False
            datas.append(rd_mol)
        data_list = [canonicalize_smiles, num_atom, num_bond, datas]
      
        all_test_data.append(data_list)
        num_valid_mol += 1
        num_valid_conf += len(datas)

    print('poster-filter: find %d molecules with %d confs' % (num_valid_mol, num_valid_conf))

    return all_test_data


def process_inst(smiles, num_atom, num_bond):
    system_prompt = f'Below is a SMILES of a molecule, generate its 3D structure. The molecule has {num_atom} atoms and {num_bond} bonds.'
    inst = '<s>[INST] <<SYS>>\n' + system_prompt + '\n<</SYS>>\n\n' + smiles + ' [/INST] '
    
    return inst


def get_mol_block(text, smiles, num_atom, num_bond):
    inst = dataset.process_inst(smiles, num_atom, num_bond).replace('<s>', '')
    mol_block = text.replace(inst, '').replace(' </s>','').replace('<s>', '')
    
    return mol_block


def process_df(data_list):
    try:
        df = pd.DataFrame(data_list, columns=['smiles', 'canonicalize_smiles', 'num_atom', 'num_bond', 'mol_block'])
        text_col = []
        for _, row in df.iterrows():
            num_atom = row['num_atom']
            num_bond = row['num_bond']
            smiles = row['canonicalize_smiles'] 
            inst = process_inst(smiles, num_atom, num_bond)
            model_answer = row['mol_block']
            text = inst + model_answer + ' </s>'
            text_col.append(text)

        df.loc[:, 'text'] = text_col
    except:
        df = pd.DataFrame(data_list, columns=['smiles', 'canonicalize_smiles', 'num_atom', 'num_bond', 'mol_block', 'smiles_index', 'conformers_index'])
    
    df['num_atom'] = df['num_atom'].astype(int)
    df['num_bond'] = df['num_bond'].astype(int)
    
    return df


def process_generated_data(data):
    data_c = copy.deepcopy(data)
    for i in range(len(data_c)):
        for j in range(len(data_c[i][4])):
            data_c[i][4][j] = get_mol_block(data_c[i][4][j], data_c[i][0], data_c[i][1], data_c[i][2])

    return data_c


if __name__ == '__main__':
    pass