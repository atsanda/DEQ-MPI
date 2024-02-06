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
import h5py

descriptors = [
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_10.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_15.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_20.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_25.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_30.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_35.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
    "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_40.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
]
for descriptor in descriptors:
    getModelForImplicitLD(descriptor, 25, save_models=True)