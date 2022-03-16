

# script that differentiates a dim_reducer model with respect to the input to help visualize what the model 'relies on'

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

# path = '/Users/rittlern/Desktop/Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/run_119/'
# save_path = '/Users/rittlern/Desktop/Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/run_119/'
#


############################################ Data Setup ##################################################

# load in the data
x_t = np.load(path + 'x_train.npy')
y_t = np.load(path + 'y_train.npy')
x_v = np.load(path + 'x_validation.npy')
y_v = np.load(path + 'y_validation.npy')

p = y_t.shape[0]  # dimension of the response
d = x_t.shape[1]  # the original dimension of the predictor space
m = int(m[0])  # the reduced dimension of the predictor space

if not synthetic_data:
    read_data_params.update({'d': d})

# set all batch sizes to the size of the training set if specified
if max_batch:
    batch_size = list(x_t.shape[0] * np.ones(num_training_phases))

# normalization (if you dim reduce with normalized data, probably you should train with normalized data too)
if normalize:
    x_t = (x_t - np.mean(x_t, axis=0)) / np.std(x_t, axis=0)

    y_t_mu, y_t_sig = np.mean(y_t, axis=1),  np.std(y_t, axis=1)
    y_t = np.transpose((np.transpose(y_t) - y_t_mu) / y_t_sig)
    np.save(save_path + 'response_scale.npy', y_t_sig)  # save so we can undo normalization on final wind plots
    np.save(save_path + 'response_location.npy', y_t_mu)

    x_v = (x_v - np.mean(x_v, axis=0)) / np.std(x_v, axis=0)
    y_v = np.transpose((np.transpose(y_v) - np.mean(y_v, axis=1)) /
                                np.std(y_v, axis=1))

# validation data to tf format
X_v = tf.Variable(np.transpose(x_v), name='Xv', dtype=tf.float32, trainable=True)
Y_v = tf.Variable(y_v, name='Yv', dtype=tf.float32, trainable=True)


# load in the model
np.random.seed(training_seed)

method = 'normal' if gaussian_assumption else est_type

if linear:
    if unique:
        fixed_part = np.reshape(np.random.normal(0, .1, size=m**2), (m, m))
        dim_reducer = FixedPartLinearDimReduce(layers, d, m, p, method, fixed_part)
    dim_reducer = LinearDimReduce(layers, d, m, p, method, seed=training_seed[0], resnet=resnet)
else:
    dim_reducer = DimReduce(layers, d, m, p, method, seed=training_seed[0], resnet=resnet)


chk_path = save_path + 'chk/'  # we always end paths with '/'
chk = tf.train.Checkpoint(dim_reducer=dim_reducer, total_step=tf.Variable(1, name='total_step'),
                          minloss=tf.Variable(1e20, name='minloss'), minloss_epoch=tf.Variable(1, name='minloss_epoch'))
chk_manager = tf.train.CheckpointManager(chk, chk_path, max_to_keep=3)
chk.restore(chk_manager.latest_checkpoint)



############################# differntiation with respect to the input ###############


with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(X_v)
    X_v = tf.transpose(X_v)
    lower_dims = tf.transpose(dim_reducer.to_lower_dims(tf.transpose(X_v)))
    loss_v = dim_reducer.loss(tf.transpose(X_v), Y_v)
    print(loss_v)

input_grads = tape.batch_jacobian(lower_dims, X_v, experimental_use_pfor=True)

input_grads_0 = input_grads[:, 0, :]   # d T(x)_0/ dx_i
input_grads_1 = input_grads[:, 1, :]   # d T(x)_1/ dx_i

## reshape back predictors back into grid
partial_grid = np.empty((m, len(predictors), X_v.shape[0], sq_size, sq_size))  # 5d tensor: (output coord, pred, sample, lat, long)
count = 0
for p in range(len(predictors)):
    for lat in range(sq_size):
        for long in range(sq_size):
            partial_grid[0, p, :, lat, long] = input_grads_0[:, count]
            partial_grid[1, p, :, lat, long] = input_grads_1[:, count]
            count += 1


partial_grid = np.abs(partial_grid)  # interested in magnitude
partial_grid_00 = np.mean(partial_grid[0, 0, :, :, :], axis=0)  # average ( dT(x)_0/d pred_0[lat,long] ) over valid set ( pred_0 = N/s for now)
partial_grid_01 = np.mean(partial_grid[0, 1, :, :, :], axis=0)  # average ( dT(x)_0/d pred_1[lat,long] ) over valid set
partial_grid_10 = np.mean(partial_grid[1, 0, :, :, :], axis=0)  # average ( dT(x)_1/d pred_0[lat,long] ) over valid set
partial_grid_11 = np.mean(partial_grid[1, 1, :, :, :], axis=0)  # average ( dT(x)_1/d pred_1[lat,long] ) over valid set
print(partial_grid_11.shape )  # should be a square


# do some plotting of the average gradient

fig, ax = plt.subplots(m, len(predictors), squeeze=False)
plt.setp(ax, xticks=[], yticks=[])
plt.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = .8, wspace = 0.4, hspace = 0.4)

ax[0, 0].set_title('$T(x)_1$ to N/S')
ax[0, 1].set_title('$T(x)_1$ to E/W')
ax[1, 0].set_title('$T(x)_2$ to N/S')
ax[1, 1].set_title('$T(x)_2$ to E/W')

im00 = ax[0, 0].imshow(partial_grid_00, interpolation='nearest')
cbar = plt.colorbar(im00, ax=ax[0, 0])
cbar.set_label('$ | \dfrac{\partial T(x)_1}{\partial \ vp_{i, j}} | $', rotation=0,  labelpad=25)

im01 = ax[0, 1].imshow(partial_grid_01, interpolation='nearest')
cbar = plt.colorbar(im01, ax=ax[0, 1])
cbar.set_label('$   | \dfrac{\partial T(x)_1}{\partial \ up_{i, j}} |$', rotation=0,  labelpad=25)

im10 = ax[1, 0].imshow(partial_grid_10, interpolation='nearest')
cbar = plt.colorbar(im10, ax=ax[1, 0])
cbar.set_label('$ | \dfrac{\partial T(x)_2}{\partial \ vp_{i, j}} |$', rotation=0,  labelpad=25)

im11 = ax[1, 1].imshow(partial_grid_11, interpolation='nearest')
cbar = plt.colorbar(im11, ax=ax[1, 1])
cbar.set_label('$  | \dfrac{\partial T(x)_2}{\partial \ up_{i, j}} |$', rotation=0,  labelpad=25)

fig.suptitle('Sensitivity of Transformation to the Inputs')
fig.savefig(path + 'sensitivity.png')


# grad is how to fuck to up T the most so in a sense it's what it keys on to make decisions
# intersting is how you see that bottom left part of the plot show in correlation plots too. it seems T_1 is for vp
# and T_0 is for up sort of

# YOU COULD AVERAGE THESE PLOTS OVER DIFFERENT INTIALIZATIONS!!!