import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.nn.functional as F

from data import *
from reconAlgos import *
from modelClasses import *
from trainerClasses import *

import gc
import time
import copy
from pathlib import Path
import h5py

def dumph5(filename, obj):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('content', data=obj)
    hf.close()

torch.manual_seed(0)
# Settings for inference:

def generate_data(psnr):
    stdScl = 0.41
    stdVal = 10**(-psnr / 20) * stdScl

    n1 = 26
    n2 = 13

    inhouseLoadStr = "inhouseData/expMatinHouse"
    sysMtxRef = loadMtxExp(inhouseLoadStr, device="cpu").reshape(-1, n1 * n2)

    testDataHR = (MRAdatasetH5NoScale("datasets/testPatches.h5", prefetch=True, device="cpu")).data
    testData, rand_scale = transformDataset(testDataHR, [26, 13], [0.5, 1], [0, 0])
    myDataGen, noise = getNoisyData(testData, stdVal, sysMtxRef, return_noise=True)
    return myDataGen, testData[:,:,:,:].squeeze()

def generate_clean_data():
    n1 = 26
    n2 = 13

    inhouseLoadStr = "inhouseData/expMatinHouse"
    sysMtxRef = loadMtxExp(inhouseLoadStr, device="cpu").reshape(-1, n1 * n2)

    testDataHR = (MRAdatasetH5NoScale("datasets/testPatches.h5", prefetch=True, device="cpu")).data
    testData, rand_scale = transformDataset(testDataHR, [26, 13], [0.5, 1], [0, 0])
    myDataGen, noise = getNoisyData(testData, 0.0, sysMtxRef, return_noise=True)
    return myDataGen, testData[:,:,:,:].squeeze()


base_folder = Path("testdata/simulated")
for psnr in [15, 25, 35]:
    psnr_folder = base_folder / f"psnr-{psnr}"
    psnr_folder.mkdir()
    meas, gt = generate_data(psnr)
    dumph5(psnr_folder / "meas.hdf5", meas)
    dumph5(psnr_folder / "gt.hdf5", gt)


psnr_folder = base_folder / "clean"
psnr_folder.mkdir()
meas, gt = generate_clean_data()
dumph5(psnr_folder / "meas.hdf5", meas)
dumph5(psnr_folder / "gt.hdf5", gt)

