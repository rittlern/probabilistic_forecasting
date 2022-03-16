
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
if restart is not None:
    save_path = restart
    path = restart
else:
    stem = 'Experiments/' if synthetic_data else 'Real_Experiments/'
    path, save_path = path_finder(stem, data_directory)

# hyperparam saving
if save_run:
    save_settings(path, save_path, data_params, training_params, synthetic_data)

############################################ Data Setup ##################################################

# load in the data
if os.path.isfile(path + 'x_train_dr.npy'):
    x_t = np.load(path + 'x_train_dr.npy')
    y_t = np.load(path + 'y_train_dr.npy')
    print('Training on separate data')
else:
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
X_v = tf.constant(np.transpose(x_v), name='Xv', dtype=tf.float32)
Y_v = tf.constant(y_v, name='Yv', dtype=tf.float32)

########################################## Some Diagnostics ##################################################
# some synthetic specific diagnostic helpers
if synthetic_data:
    mu, Sigma = load_gen_params(path, generation_type, p)


# compute the optimal transformation to p space from normal theory
Sigma = np.cov(np.concatenate((y_t, np.transpose(x_t)), axis=0)) if estimate_var_baseline else Sigma
trans_normal_opt = np.transpose(np.linalg.solve(Sigma[p:, p:], Sigma[p:, :p]))
var_opt = analytic_transformation_variance(trans_normal_opt, Sigma, p)


# set up non-diagonal gaussian kernels, if applicable
if est_type == 'kde_gen':  # using a non-diagonal kernel
    A_inv = np.eye(m + p) if not isinstance(A, np.ndarray) else np.linalg.inv(A)
    assert A_inv.shape[0] == (m + p), 'Kernel covariance matrix must match dimension of data.'

#print(loss_baseline_gaussian_knn_mc(mu, Sigma, k_yz[0], k_z[0],normalize, p, 10, batch_size[0]))
#print(loss_baseline_gaussian_knn_mc(mu, Sigma, k_yz[0], k_z[0],normalize, p, 10, y_v.shape[1]))

############################################### TRAINING ##################################################

np.random.seed(training_seed)

losses_t, losses_v, vars_ = list(), list(), list()  # store true variances of y|T(x) when training on synthetic data

method = 'normal' if gaussian_assumption else est_type

if linear:
    if unique:
        fixed_part = np.reshape(np.random.normal(0, .1, size=m**2), (m, m))
        dim_reducer = FixedPartLinearDimReduce(layers, d, m, p, method, fixed_part)
    dim_reducer = LinearDimReduce(layers, d, m, p, method, seed=training_seed[0], resnet=resnet)
else:
    dim_reducer = DimReduce(layers, d, m, p, method, seed=training_seed[0], resnet=resnet)

# get trainable variables into a flat list
tvs_temp = [layer.trainable_variables for layer in layers]
tvs = [var for sub in tvs_temp for var in sub]


# Some checkpoint management
restart = 0
chk_path = save_path + 'chk/'  # we always end paths with '/'
chk = tf.train.Checkpoint(dim_reducer=dim_reducer, total_step=tf.Variable(1, name='total_step'),
                          minloss=tf.Variable(1e20, name='minloss'), minloss_epoch=tf.Variable(1, name='minloss_epoch'))
chk_manager = tf.train.CheckpointManager(chk, chk_path, max_to_keep=1)
if restart is False:
    first_step = 0
    minloss_global = sys.float_info.max
else:
    chk.restore(chk_manager.latest_checkpoint)
    first_step = chk.total_step.numpy()
    minloss_global = chk.minloss.numpy()
    minloss_epoch = first_step


#########################

# loop over training phases
for phase in training_phases:

    print(" \nLearning dimension reduction in training phase " + str(phase) + '...\n ')

    # set the optimization hyperparameters for this training phase
    num_steps_phase, learning_rate_phase = num_steps[phase], learning_rate[phase]
    batch_size_phase, num_batches_to_accum_phase = int(batch_size[phase]), int(num_batches_to_accum[phase])

    # update estimation hyperparameters for the phase
    h_phase = h[phase] if isinstance(h[phase], np.ndarray) else np.full(m + p, h[phase])
    if len(h_phase) != (m+p):
        raise ValueError('Number of smoothing parameters should be 1, or equal to number of dims.')

    hyperparameter_dict = {'batch_size': batch_size_phase, 'k_yz': k_yz[phase], 'k_z': k_z[phase], 'h': h_phase}
    dim_reducer.update_hyperparams(hyperparameter_dict)

    # (re-)define optimizer based on phase parameters
    opz = tf.optimizers.Adam(learning_rate_phase / num_batches_to_accum_phase)

    for step in range(num_steps_phase):

        accum_grads = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in tvs]  # container for grad accumulation

        for batch in range(num_batches_to_accum_phase):

            t_idx = np.random.choice(range(x_t.shape[0]), batch_size_phase, replace=False)
            X_t = tf.constant(np.transpose(x_t[t_idx, :]), name='X', dtype=tf.float32)
            Y_t = tf.constant(y_t[:, t_idx], name='Y', dtype=tf.float32)

            with tf.GradientTape(watch_accessed_variables=True) as tape:

                loss_t = dim_reducer.loss(X_t, Y_t)

                if batch == 0:  # get the loss with the current parameter settings, before we apply gradient
                    loss_v = dim_reducer.loss(X_v, Y_v)

                batch_grads = tape.gradient(loss_t, tvs)
                accum_grads = [accum_grads[i].assign_add(grad) for i, grad in enumerate(batch_grads)]

        opz.apply_gradients(zip(accum_grads, tvs))

        # training diagnostics/documentation
        losses_t.append(loss_t.numpy())
        losses_v.append(loss_v.numpy())

        if loss_t.numpy() < minloss_global:  #used to be loss_v but im going for independence from validation set
            minloss_global = loss_v.numpy()
            chk.minloss.assign(minloss_global)
            minloss_epoch = chk.total_step
            chk_manager.save()

        print("Iteration: {};  Training loss: {};  Validation Loss: {}".format(chk.total_step.numpy(),
                                                                               np.around(loss_t.numpy(), 5),
                                                                                   np.around(loss_v.numpy(), 5)))

        # if the learned transformation is linear and synth data used, we can compute analytical conditional variance
        if isinstance(dim_reducer, (LinearDimReduce, FixedPartLinearDimReduce)):
            T = dim_reducer.current_transform()

            if save_run:
                np.save(save_path + 'T' + str(chk.total_step.numpy()) + '.npy', T)
            if synthetic_data:
                if chk.total_step.numpy() % 10 == 0 and chk.total_step.numpy() != 0:
                    var_trans = analytic_transformation_variance(T, Sigma, p)
                    vars_.append(var_trans)
                    print_variances(True, var_trans, var_opt)

        chk.total_step.assign_add(1)


##################################### POST TRAINING ###################################################

# do some plotting of the training
if synthetic_data:
    loss_plot(losses_t, losses_v,
              save_path, first_step, chk.total_step.numpy(), training_params, data_params)
else:
    loss_plot(losses_t, losses_v,
              save_path, first_step, chk.total_step.numpy(), training_params, read_data_params)


# save transformed data (useful for nonlinear transforms)
chk.restore(chk_manager.latest_checkpoint).assert_consumed()

if os.path.isfile(path + 'x_train_dr.npy'):  # we need to load in the other training data to transform in this case
    x_t = np.load(path + 'x_train.npy')
    y_t = np.load(path + 'y_train.npy')

    if normalize:
        x_t = (x_t - np.mean(x_t, axis=0)) / np.std(x_t, axis=0)
        y_t = np.transpose((np.transpose(y_t) - np.mean(y_t, axis=1)) / np.std(y_t, axis=1))


X_t = tf.constant(np.transpose(x_t), name='X', dtype=tf.float32)

v_t = np.concatenate((y_t, dim_reducer.to_lower_dims(X_t).numpy()), axis=0)
v_v = np.concatenate((y_v, dim_reducer.to_lower_dims(X_v).numpy()), axis=0)

x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')
x_test = (x_test - np.mean(x_v, axis=0)) / np.std(x_test, axis=0)
y_test = np.transpose((np.transpose(y_test) - np.mean(y_test, axis=1)) /
                                np.std(y_test, axis=1))
X_test = tf.constant(np.transpose(x_test), name='X_test', dtype=tf.float32)
v_test = np.concatenate((y_test, dim_reducer.to_lower_dims(X_test).numpy()), axis=0)

np.save(save_path + 'v_t.npy', v_t)
np.save(save_path + 'v_v.npy', v_v)
np.save(save_path + 'v_test.npy', v_test)

# clean up the models we no longer need, and do some further saving
if save_run:
    if isinstance(dim_reducer, (LinearDimReduce, FixedPartLinearDimReduce)):
        T = clean_up(losses_v, save_path, first_step, chk.total_step.numpy())

    if synthetic_data:
        var_trans = analytic_transformation_variance(T, Sigma, p)
        with open(save_path + "/variance_report.txt", "w") as file_:
            file_.write("Var(y|T(x)) for T with lowest validation error: {}".format(var_trans))
            file_.write("\n")
            file_.write("Var(y|T^*(x)), i.e. variance under sufficient dim reduction: {}".format(var_opt))
else:
    shutil.rmtree(save_path)


print('\n')
print('Training complete. Results can be found in {}'.format(save_path))
print('\n')
