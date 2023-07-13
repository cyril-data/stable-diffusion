import numpy as np
import argparse

import os, glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--path', type=str,
                        default='data/train_IS_1_1.0_0_0_0_0_0_256_done_red/mean_with_orog.npy'
                        # default='/scratch/mrmn/reganc/data/1024/'
                        )
    arg = parser.parse_args()


    path = arg.path

    means = []
    vars = []
    grids=[]
    for filename in glob.glob(os.path.join(path, '*.npy')):
        grid = np.load(os.path.join(os.getcwd(), filename))
        grids.append(grid)
        # np.concatenate((grid, np.load(os.path.join(os.getcwd(), filename))), axis=0)

        plt.imshow(grid[0,:,:], cmap='viridis')
        plt.colorbar()
        plt.close()
        plt.imshow(grid[0,:,:], cmap='viridis')
        plt.colorbar()
        plt.close()
        plt.imshow(grid[0,:,:], cmap='RdBu_r')
        plt.colorbar()
        plt.close()


    grid = np.array(grids)


    print(f"path {arg.path} : \n{grid.shape}")

    print(f"u \t: mean = {np.mean(grid[:,0,:,:])}, var = {np.var(grid[:,0,:,:])}")
    print(f"v \t: means : {np.mean(grid[:,1,:,:])}, var = {np.var(grid[:,1,:,:])}")
    print(f"t2m\t: means : {np.mean(grid[:,2,:,:])}, var = {np.var(grid[:,2,:,:])}")
