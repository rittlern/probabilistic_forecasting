

# file for implementing dim_reduce in a more organized, oop fashion

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from DimReduceNet import *
import shutil

# get parameters
from dim_reduce_job import training_params, num_training_phases
from generate_data_job import data_params
from read_data_job import read_data_params
from dim_optimality_job import dim_op_params

########################################## Parameter/Saving Setup ##################################################

# Insert the current directory in the python search path
sys.path.insert(0, ".")

# parameter setup
for param in training_params.keys():
    if (type(training_params[param])) is int or (type(training_params[param])) is float or \
            isinstance(training_params[param], np.ndarray):
        training_params[param] = [training_params[param] for i in range(num_training_phases)]
globals().update(training_params)
globals().update(dim_op_params)

if synthetic_data:
    globals().update(data_params)
else:
    globals().update(read_data_params)

# find the correct paths for loading and saving
stem = 'Experiments/' if synthetic_data else 'Real_Experiments/'
path, save_path = path_finder(stem, data_directory)
if restart:
    save_path = restart

# hyperparam saving
if save_run:
    save_settings(path, save_path, data_params, training_params, synthetic_data)

np.random.seed(training_seed)
tf.random.set_seed(training_seed)

##################


# load in the data
if os.path.isfile(path + 'x_train_dr.npy'):
    x_t_dr = np.load(path + 'x_train_dr.npy')
    y_t_dr = np.load(path + 'y_train_dr.npy')
    print('Training on separate data')

x_t = np.load(path + 'x_train.npy')  # training data for the joint modeling
y_t = np.load(path + 'y_train.npy')

x_v = np.load(path + 'x_validation.npy')
y_v = np.load(path + 'y_validation.npy')

x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')

p = y_t.shape[0]  # dimension of the response
d = x_t.shape[1]  # the original dimension of the predictor space
m = int(m[0])  # the reduced dimension of the predictor space

if normalize:
    if os.path.isfile(path + 'x_train_dr.npy'):
        x_t_dr = (x_t_dr - np.mean(x_t_dr, axis=0)) / np.std(x_t_dr, axis=0)
        y_t_dr = np.transpose((np.transpose(y_t_dr) - np.mean(y_t_dr, axis=1)) /
                           np.std(y_t_dr, axis=1))

    x_t = (x_t - np.mean(x_t, axis=0)) / np.std(x_t, axis=0)
    y_t_mu, y_t_sig = np.mean(y_t, axis=1),  np.std(y_t, axis=1)
    y_t = np.transpose((np.transpose(y_t) - y_t_mu) / y_t_sig)
    np.save(save_path + 'response_scale.npy', y_t_sig)  # save so we can undo normalization on final wind plots
    np.save(save_path + 'response_location.npy', y_t_mu)

    x_v = (x_v - np.mean(x_v, axis=0)) / np.std(x_v, axis=0)
    y_v = np.transpose((np.transpose(y_v) - np.mean(y_v, axis=1)) /
                                np.std(y_v, axis=1))

    x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
    y_test = np.transpose((np.transpose(y_test) - np.mean(y_test, axis=1)) /
                       np.std(y_test, axis=1))

# PCA reduction
if naive_method == 'pca':  # https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

    if os.path.isfile(path + 'x_train_dr.npy'):
        u, s, v = np.linalg.svd(x_t_dr, full_matrices=True)
        x_t_proj = np.matmul(v, np.transpose(x_t))  # project joint model training data

    else:
        u, s, v = np.linalg.svd(x_t, full_matrices=True)
        x_t_proj = np.matmul(u[:, :m], np.diag(s[:m]))  # no need to project training data in this case

    # check: https://stackoverflow.com/questions/55441022/how-to-aply-the-same-pca-to-train-and-test-set

    x_v_proj = np.matmul(v, np.transpose(x_v))  # project validation onto SVD dictated space
    x_test_proj = np.matmul(v, np.transpose(x_test))

    v_t = np.concatenate((np.transpose(y_t), np.transpose(x_t_proj[:m, :])), axis=1)  # concat the lower dimensional representation, response
    v_v = np.concatenate((np.transpose(y_v), np.transpose(x_v_proj[:m, :])), axis=1)
    v_test = np.concatenate((np.transpose(y_test), np.transpose(x_test_proj[:m, :])), axis=1)


# in case we want to take the top k predictors (should only use training set anyway to pick predictors)
if naive_method == 'top_k':

    m = m if m % 2 == 0 else m - 1  # assuming two responses
    best = np.zeros((len(predictors), int(m/2), 2))  # store the count of the most corr predictor so far for each pred

    v_t = np.transpose(y_t)
    v_v = np.transpose(y_v)
    v_test = np.transpose(y_test)

    for p in range(len(predictors)):
            for i in range(x_t.shape[1]):
                if os.path.isfile(path + 'x_train_dr.npy'):
                    corr_with_p = np.abs(
                        np.corrcoef(y_t_dr[p, :], x_t_dr[:, i])[1, 0])  # these are all lower as here just train set
                else:
                    corr_with_p = np.abs(
                        np.corrcoef(y_t[p, :], x_t[:, i])[1, 0])  # these are all lower as here just train set

                if np.sum(best[p, :, 1] < corr_with_p) > 0:  # then we have a new best grid point
                    best[p, -1, :] = np.array((i, corr_with_p))  # replace the lowest corr coff with p
                    best[p, :, :] = best[p, best[p, :, 1].argsort()]

            for b in range(int(m / 2)):
                data_idx = int(best[p, b, 0])

                v_t = np.concatenate((v_t, np.transpose(x_t[:, data_idx].reshape((1, x_t.shape[0])))), axis=1)
                v_v = np.concatenate((v_v, x_v[:, data_idx].reshape((x_v.shape[0], 1))), axis=1)
                v_test = np.concatenate((v_test, x_test[:, data_idx].reshape((x_test.shape[0], 1))), axis=1)

# save the data
np.save(path + 'v_t.npy', np.transpose(v_t))
np.save(path + 'v_v.npy', np.transpose(v_v))
np.save(path + 'v_test.npy', np.transpose(v_test))
