import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--npy_file', type=str,
                        default='data/train_IS_1_1.0_0_0_0_0_0_256_done_red/mean_with_orog.npy'
                        # default='/scratch/mrmn/reganc/data/1024/'
                        )
    arg = parser.parse_args()

    grid = np.load(arg.npy_file)

    print(f"file {arg.npy_file} : \n{grid.shape}\n{grid}")
