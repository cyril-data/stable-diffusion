#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet class from Importance_Sampled images

"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch


def print_stat(image, inc="=", depht=2, type="numpy", tranpose=True):

    if type == "numpy":
        if tranpose:
            print("numpy tranpose image", image.shape)
            image = np.transpose(image, axes=[2, 1, 0])
            print("numpy tranpose image 2", image.shape)

        print("\n"+inc*depht +
              f" grid u  \t m {np.mean(image[0,:,:])},\t s {np.std(image[0,:,:])},\t min  {np.min(image[0,:,:])},\t max {np.max(image[0,:,:])}")
        print(inc*depht +
              f" grid v \t m {np.mean(image[1,:,:])},\t s {np.std(image[1,:,:])},\t min  {np.min(image[1,:,:])},\t max {np.max(image[1,:,:])}")
        print(inc*depht +
              f" grid t \t m {np.mean(image[2,:,:])},\t s {np.std(image[2,:,:])},\t min  {np.min(image[2,:,:])},\t max {np.max(image[2,:,:])}")
    else:
        if tranpose:
            image = torch.transpose(image, 0, 2)
        print("\n"+inc*depht +
              f" grid u  \t m {torch.mean(image[0,:,:])},\t s {torch.std(image[0,:,:])},\t min  {torch.min(image[0,:,:])},\t max {torch.max(image[0,:,:])}")
        print(inc*depht +
              f" grid v \t m {torch.mean(image[1,:,:])},\t s {torch.std(image[1,:,:])},\t min  {torch.min(image[1,:,:])},\t max {torch.max(image[1,:,:])}")
        print(inc*depht +
              f" grid t \t m {torch.mean(image[2,:,:])},\t s {torch.std(image[2,:,:])},\t min  {torch.min(image[2,:,:])},\t max {torch.max(image[2,:,:])}")


# reference dictionary to know what variables to sample where
# do not modify unless you know what you are doing

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4}

################


class ISDataset(Dataset):

    def __init__(self, data_dir, ID_file, var_indexes, crop_indexes,
                 add_coords=False):

        self.data_dir = data_dir
        self.labels = pd.read_csv(data_dir+ID_file)

        # portion of data to crop from (assumed fixed)

        self.CI = crop_indexes
        self.VI = var_indexes
        # self.coef_avg2D = coef_avg2D

        # adding 'positional encoding'
        self.add_coords = add_coords
        Means = np.load(data_dir+'mean_with_orog.npy')[self.VI]
        Maxs = np.load(data_dir+'max_with_orog.npy')[self.VI]
        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # idx=idx+19
        sample_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        sample = np.float32(np.load(sample_path+'.npy')
                            )[self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]]
        # print("where I am", len(sample))
        importance = self.labels.iloc[idx, 1]
        position = self.labels.iloc[idx, 2]

        # transpose to get off with transform.Normalize builtin transposition
        sample = sample.transpose((1, 2, 0))

        # sample[:,:,2]=2.*(sample[:,:,2]-251.14634704589844)/(315.44622802734375-251.14634704589844)-1.
        # sample[:,:,0]=2.*(sample[:,:,0]+27.318836212158203)/(29.181968688964844 + 27.318836212158203)-1.
        # sample[:,:,1]=2.*(sample[:,:,1]+25.84168815612793)/(27.698963165283203 + 25.84168815612793)-1.

        self.transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
                transforms.Normalize([-1, -1, -1], [2, 2, 2]),
                # transforms.Lambda(lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=self.coef_avg2D,
                #                                                           stride=self.coef_avg2D)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Normalize(
                #     [0.5 for _ in range(config.CHANNELS_IMG)],
                #     [0.5 for _ in range(config.CHANNELS_IMG)],
                # ),
            ]
        )

        # TransM0 = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        # TransM1 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(self.means, self.stds),
        # ])

        # TransM2 = transforms.Compose([
        #     transforms.Normalize([-1, -1, -1], [2, 2, 2]),
        # ])

        # invTransM1 = transforms.Compose([
        #     transforms.Normalize(
        #         mean=[0.] * 3, std=[1 / el for el in self.stds]),
        #     transforms.Normalize(
        #         mean=[-el for el in self.means], std=[1.] * 3),
        # ])

        # invTransM2 = transforms.Compose([
        #     transforms.Normalize(
        #         mean=[0.] * 3, std=[1 / el for el in [2, 2, 2]]),
        #     transforms.Normalize(
        #         mean=[-el for el in [-1, -1, -1]], std=[1.] * 3),
        # ])

        # invTrans = transforms.Compose([
        #     transforms.Normalize(
        #         mean=[0.] * 3, std=[1 / el for el in [2, 2, 2]]),
        #     transforms.Normalize(
        #         mean=[-el for el in [-1, -1, -1]], std=[1.] * 3),
        #     transforms.Normalize(
        #         mean=[0.] * 3, std=[1 / el for el in self.stds]),
        #     transforms.Normalize(
        #         mean=[-el for el in self.means], std=[1.] * 3),

        # ])

        # test0 = sample.copy()

        # # test1 = test0.copy()
        # test1 = TransM1(test0)

        # # test2 = test1.detach().clone()
        # test2 = TransM2(test1)
        # # print_stat(test2, inc="--", depht=2,
        # #            type="tensor", tranpose=False)

        # # invtest2 = test2.detach().clone()
        # invtest2 = invTransM2(test2)

        # MAE = torch.mean(
        #     torch.abs(invtest2 - test1))
        # print("MAE test2 :", MAE.item())

        # # invtest1 = invtest2.detach().clone()
        # invtest1 = invTransM1(invtest2)

        # print("test0", test0.shape)
        # print("TransM0(test0)", TransM0(test0).shape)

        # MAE = torch.mean(
        #     torch.abs(invtest1 - TransM0(test0)))
        # print("MAE test1 :", MAE.item())

        # invtest = invTrans(test2)
        # MAE = torch.mean(
        #     torch.abs(invtest - TransM0(test0)))
        # print("MAE test :", MAE.item())

        # sample_init = sample.copy()

        # sample_verif = invTransM1(sample)

        # sample_verif = invTransM2(sample_verif)

        # print("sample_verif invTransM2", sample_verif.shape)
        # print("sample_init", sample_init.shape)
        # print_stat(sample_verif, inc="**", depht=2,
        #            type="tensor", tranpose=False)

        # print_stat(sample_init, inc="--", depht=2, type="numpy")

        # MAE = torch.mean(
        #     torch.abs(torch.from_numpy(np.transpose(sample_init, axes=[2, 1, 0])) - sample_verif))

        # print("MAE M1 :", MAE.item())

        # print("Handle 3", sample.shape)
        # print("sample", sample.shape)
        # T1 = sample
        # print("sample trans", torch.transpose(sample, 0, 2).shape)
        # print("sample trans *2", torch.transpose(
        #     torch.transpose(sample, 0, 2), 0, 2).shape)

        # T2 = torch.transpose(
        #     torch.transpose(sample, 0, 2), 0, 2)

        # print("torch.eq(T1, T2)", torch.equal(T1, T2))

        sample = self.transform(sample)

        # print("sample max 0", torch.max(sample))
        # # test 0 learn:
        # sample =  torch.minimum(sample, torch.zeros_like(sample))

        # print("sample max 1", torch.max(sample))

        return torch.transpose(sample, 0, 2), importance, position


class ISData_Loader_train(ISDataset):
    def __init__(self,
                 batch_size=1,
                 var_indexes=[1, 2, 3],
                 crop_indexes=[78, 206, 55, 183],
                 path="data/train_IS_1_1.0_0_0_0_0_0_256_done_red/",
                 shuf=False,
                 add_coords=False,
                 num_workers=0):

        super().__init__(path, 'IS_method_labels.csv',
                         var_indexes, crop_indexes)

        self.path = path
        self.batch = batch_size
        if num_workers == 0:
            num_workers = self.batch*2
        self.shuf = shuf  # shuffle performed once per epoch
        self.VI = var_indexes
        self.CI = crop_indexes

        Means = np.load(path+'mean_with_orog.npy')[self.VI]
        Maxs = np.load(path+'max_with_orog.npy')[self.VI]

        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))
        self.add_coords = add_coords

    def loader(self):
        dataset = ISDataset(self.path, 'IS_method_labels.csv',
                            self.VI, self.CI)

        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            num_workers=self.batch*2,
                            pin_memory=False,
                            shuffle=True,
                            drop_last=True,
                            )
        return loader, dataset

    def norm_info(self):
        return self.means, self.stds


class ISData_Loader_val(ISDataset):
    def __init__(self,
                 batch_size=1,
                 var_indexes=[1, 2, 3],
                 crop_indexes=[78, 206, 55, 183],
                 path="data/train_IS_1_1.0_0_0_0_0_0_256_done_red/",
                 shuf=False,
                 add_coords=False):

        super().__init__(path, 'IS_method_labels.csv',
                         var_indexes, crop_indexes)

        self.path = path
        self.batch = batch_size
        self.shuf = shuf  # shuffle performed once per epoch
        self.VI = var_indexes
        self.CI = crop_indexes

        Means = np.load(path+'mean_with_orog.npy')[self.VI]
        Maxs = np.load(path+'max_with_orog.npy')[self.VI]

        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))
        self.add_coords = add_coords

    def _prepare(self):
        ISDataset(self.path, 'IS_method_labels.csv',
                  self.VI, self.CI)

    def loader(self):
        dataset = ISDataset(self.path, 'IS_method_labels.csv',
                            self.VI, self.CI)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            num_workers=1,
                            pin_memory=False,
                            shuffle=True,
                            drop_last=True,
                            )
        return loader, dataset

    def norm_info(self):
        return self.means, self.stds


# class ImageNetSRTrain(ImageNetSR):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def get_base(self):
#         with open("data/imagenet_train_hr_indices.p", "rb") as f:
#             indices = pickle.load(f)
#         dset = ImageNetTrain(process_images=False,)
#         return Subset(dset, indices)


# class ImageNetSRValidation(ImageNetSR):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def get_base(self):
#         with open("data/imagenet_val_hr_indices.p", "rb") as f:
#             indices = pickle.load(f)
#         dset = ImageNetValidation(process_images=False,)
#         return Subset(dset, indices)
