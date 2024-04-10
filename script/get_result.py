import argparse
import pickle

from conf3d import utils, dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of COV score')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        generated_data = pickle.load(f)

    generated_data = dataset.process_generated_data(generated_data)
    