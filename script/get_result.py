import argparse
import os
import glob
import pickle
import statistics
import sys
sys.path.append('/hpc2hdd/home/yli106/smiles2mol')
from conf3d import utils
from tqdm import tqdm
from rdkit import RDLogger

# python -u /hpc2hdd/home/yli106/smiles2mol/script/get_result.py --input /hpc2hdd/home/yli106/smiles2mol/GEOM/generated --threshold 0.5  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of COV score')
    args = parser.parse_args()

    
    # with open(args.input, 'rb') as f:
    #     generated_data = pickle.load(f)

    generated_data = []
    file_list = glob.glob(os.path.join(args.input, 'compare.pkl'))
    for file in file_list:
        with open(file, 'rb') as f:
            generated_data_p = pickle.load(f)
        generated_data.extend(generated_data_p)
    # utils.add_element(generated_data)
    
    test_generated_data = []
    for i in range(len(generated_data)):
        if len(generated_data[i]) == 5:#and len(generated_data[i][4])/len(generated_data[i][3])>=2
            test_generated_data.append([generated_data[i][3],generated_data[i][4]])

    print(len(test_generated_data))
    cov_list, mat_list,cov_p_list, mat_p_list = [], [], [], []
    valid_count = 0
    all_count = 0
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    for i in tqdm(range(len(test_generated_data))):
        ref_list = test_generated_data[i][0]
        gen_list = utils.filter_gen_list(test_generated_data[i][1],ref_list)
        valid_count += len(gen_list)
        all_count += len(ref_list)
        cov, mat = utils.get_cov_mat(gen_list, ref_list, threshold=args.threshold)
        cov_list.append(cov)
        mat_list.append(mat)
        cov_p, mat_p = utils.get_cov_mat_p(gen_list, ref_list, threshold=args.threshold)
        cov_p_list.append(cov_p)
        mat_p_list.append(mat_p)
    
    cov_list = list(filter(lambda x: x is not None, cov_list))
    mat_list = list(filter(lambda x: x is not None, mat_list))
    cov_p_list = list(filter(lambda x: x is not None, cov_p_list))
    mat_p_list = list(filter(lambda x: x is not None, mat_p_list))
    
    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f\n' \
        %(statistics.mean(cov_list), statistics.median(cov_list), statistics.mean(mat_list), statistics.median(mat_list)))
    print('Coverage-Precision Mean: %.4f | Coverage-Precision Median: %.4f | Match-Precision Mean: %.4f | Match-Precision Median: %.4f' \
        %(statistics.mean(cov_p_list), statistics.median(cov_p_list), statistics.mean(mat_p_list), statistics.median(mat_p_list)))
    print('Valid: %.4f' %(50*valid_count/all_count))
    print('Available: %.4f' %(100*len(cov_list)/len(test_generated_data)))


    cov_list, mat_list,cov_p_list, mat_p_list = [], [], [], []
    valid_count = 0
    all_count = 0
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    for i in tqdm(range(len(test_generated_data))):
        ref_list = test_generated_data[i][0]
        gen_list = utils.filter_gen_list(test_generated_data[i][1],ref_list)
        valid_count += len(gen_list)
        all_count += len(ref_list)
        cov, mat = utils.get_cov_mat(gen_list, ref_list, threshold=args.threshold, useFF=True)
        cov_list.append(cov)
        mat_list.append(mat)
        cov_p, mat_p = utils.get_cov_mat_p(gen_list, ref_list, threshold=args.threshold)
        cov_p_list.append(cov_p)
        mat_p_list.append(mat_p)
    
    cov_list = list(filter(lambda x: x is not None, cov_list))
    mat_list = list(filter(lambda x: x is not None, mat_list))
    cov_p_list = list(filter(lambda x: x is not None, cov_p_list))
    mat_p_list = list(filter(lambda x: x is not None, mat_p_list))

    print('\nuseFF')
    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f\n' \
        %(statistics.mean(cov_list), statistics.median(cov_list), statistics.mean(mat_list), statistics.median(mat_list)))
    print('Coverage-Precision Mean: %.4f | Coverage-Precision Median: %.4f | Match-Precision Mean: %.4f | Match-Precision Median: %.4f' \
        %(statistics.mean(cov_p_list), statistics.median(cov_p_list), statistics.mean(mat_p_list), statistics.median(mat_p_list)))
    print('Valid: %.4f' %(50*valid_count/all_count))
    print('Available: %.4f' %(100*len(cov_list)/len(test_generated_data)))