import argparse
import pickle
import statistics
from conf3d import utils, dataset
from tqdm import tqdm

# python -u /hpc2hdd/home/yli106/smiles2mol/script/get_result.py --input /hpc2hdd/home/yli106/smiles2mol/GEOM/generated/inference_llama2_7b_chat.pkl --threshold 0.5  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of COV score')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        generated_data = pickle.load(f)

    cov_list, mat_list,cov_p_list, mat_p_list = [], [], [], []
    for i in tqdm(range(len(generated_data))):
        gen_list = utils.filter_gen_list(generated_data[i][4])
        ref_list = generated_data[i][3]
        cov, mat = utils.get_cov_mat(gen_list, ref_list, threshold=args.threshold)
        cov_p, mat_p = utils.get_cov_mat_p(gen_list, ref_list, threshold=args.threshold)
        cov_list.append(cov)
        mat_list.appen(mat)
        cov_p_list.append(cov_p)
        mat_p_list.append(mat_p)
    
    cov_list = list(filter(lambda x: x is not None, cov_list))
    cov_p_list = list(filter(lambda x: x is not None, cov_p_list))
    mat_list = list(filter(lambda x: x is not None, mat_list))
    mat_p_list = list(filter(lambda x: x is not None, mat_p_list))
    
    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f\n' \
        %(statistics.mean(cov_list), statistics.median(cov_list), statistics.mean(mat_list), statistics.median(mat_list)))
    print('Coverage-Precision Mean: %.4f | Coverage-Precision Median: %.4f | Match-Precision Mean: %.4f | Match-Precision Median: %.4f' \
        %(statistics.mean(cov_p_list), statistics.median(cov_p_list), statistics.mean(mat_p_list), statistics.median(mat_p_list)))