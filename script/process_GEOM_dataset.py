import os
import argparse
import pandas as pd
import pickle
import sys
sys.path.append('/hpc2hdd/home/yli106/smiles2mol')
from conf3d import dataset

# python /hpc2hdd/home/yli106/smiles2mol/script/process_GEOM_dataset.py --base_path /hpc2hdd/home/yli106/smiles2mol/GEOM --dataset_name qm9 --confmin 50 --confmax 500
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--dataset_name', type=str, choices=['qm9', 'drugs'])
    parser.add_argument('--tot_mol_size', type=int, default=50000)
    parser.add_argument('--conf_per_mol', type=int, default=5)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--test_mol_size', type=int, default=200)
    parser.add_argument('--confmin', type=int, default=50)
    parser.add_argument('--confmax', type=int, default=500)            
    args = parser.parse_args()
    rdkit_folder_path = os.path.join(args.base_path, 'rdkit_folder')

    train_data, val_data, test_data, index2split = dataset.preprocess_GEOM_dataset(rdkit_folder_path, args.dataset_name, \
                                        conf_per_mol=args.conf_per_mol, train_size=args.train_size, \
                                        tot_mol_size=args.tot_mol_size, seed=2021)

    processed_data_path = os.path.join(args.base_path, '%s_processed' % args.dataset_name)
    os.makedirs(processed_data_path, exist_ok=True)

    # save train and val data
    df =dataset.process_df(train_data)
    df.to_csv(os.path.join(processed_data_path, 'train_data_%dk.csv' % ((len(train_data) // args.conf_per_mol) // 1000)),index=False)
    print('save train %dk done' % ((len(train_data) // args.conf_per_mol) // 1000))

    df = dataset.process_df(val_data)
    df.to_csv(os.path.join(processed_data_path, 'val_data_%dk.csv' % ((len(val_data) // args.conf_per_mol) // 1000)),index=False)
    print('save val %dk done' % ((len(val_data) // args.conf_per_mol) // 1000))
    del test_data

    # filter test data
    test_data = dataset.get_GEOM_testset(rdkit_folder_path, args.dataset_name, block=[train_data, val_data], \
                                         tot_mol_size=args.test_mol_size, seed=2021, \
                                         confmin=args.confmin, confmax=args.confmax)
    with open(os.path.join(processed_data_path, 'test_data_%d.pkl' % (args.test_mol_size)), "wb") as fout:
        pickle.dump(test_data, fout)
    print('save test %d done' % (args.test_mol_size))