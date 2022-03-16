###########################  TF 1.x DIM REDUCTION #############################################


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from dim_reduce_v1_functions import *
import shutil

# get parameters
from dim_reduce_v1_job import training_params, num_training_phases
from generate_data_job import data_params
from read_data_job import read_data_params
from dim_optimality_job import dim_op_params

############################################ SETUP ####################################################

# Insert the current directory in the python search path
sys.path.insert(0, ".")

# suppress depreciation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# disable eager execution so we can run on computation graphs
tf.compat.v1.disable_eager_execution()

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

# param saving
if save_run:
    save_settings(path, save_path, data_params, training_params, synthetic_data)

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

if max_batch:  # set all batch sizes to the size of the training set
    batch_size = list(x_t.shape[0] * np.ones(num_training_phases))

if normalize:  # batches will be normalized, so normalize validation set too
    x_v = (x_v - np.mean(x_v, axis=0)) / np.std(x_v, axis=0)
    y_v = np.transpose((np.transpose(y_v) - np.mean(y_v, axis=1)) /
                                np.std(y_v, axis=1))

# some synthetic specific diagnostic helpers
if synthetic_data:
    mu, Sigma = load_gen_params(path, generation_type, p)

# settings sanity checks
#assert estimate_var_baseline or synthetic_data, 'Cannot use true baseline variance given real data...'
#assert not (true_cov and not synthetic_data), 'Cannot use the true covariance in training on real data...'

# compute the optimal transformation to p space from normal theory
Sigma = np.cov(np.concatenate((y_t, np.transpose(x_t)), axis=0)) if estimate_var_baseline else Sigma
trans_normal_opt = np.transpose(np.linalg.solve(Sigma[p:, p:], Sigma[p:, :p]))
var_opt = analytic_transformation_variance(trans_normal_opt, Sigma, p)
init_opt = trans_normal_opt if opt_init else None

# set up non-diagonal gaussian kernels, if applicable
if est_type == 'kde_gen':  # using a non-diagonal kernel
    A_inv = np.eye(m + p) if not isinstance(A, np.ndarray) else np.linalg.inv(A)
    assert A_inv.shape[0] == (m + p), 'Kernel covariance matrix must match dimension of data.'

#print(loss_baseline_gaussian_knn_mc(mu, Sigma, k_yz[0], k_z[0],normalize, p, 10, batch_size[0]))
#print(loss_baseline_gaussian_knn_mc(mu, Sigma, k_yz[0], k_z[0],normalize, p, 10, y_v.shape[1]))
############################################### TRAINING ##################################################

np.random.seed(training_seed)
losses_t = list()
losses_v = list()
vars_ = list()  # store true variances of y|T(x) when training on synthetic data
first_step = 0
total_step = 0

# define computation graph over each training phase
for phase in training_phases:
    print("Optimizing in training phase " + str(phase) + '...')

    # set the parameters for this training phase
    num_steps_phase = num_steps[phase]
    learning_rate_phase = learning_rate[phase]
    batch_size_phase = int(batch_size[phase])
    num_batches_to_accum_phase = int(num_batches_to_accum[phase])
    k_yz_phase = k_yz[phase]
    k_z_phase = k_z[phase]
    h_phase = h[phase] if isinstance(h[phase], np.ndarray) else np.full(m+p, h[phase])
    assert len(h_phase) == (m+p), 'Number of smoothing parameters should be 1, or equal to number of dims.'

    # initialize the phase
    # tf.reset_default_graph()
    params, fixed, fs = set_T(phase, training_phases, m, d, p, training_seed, unique=unique,
                                     resume_model=resume_model, resume_model_dir=resume_model_dir,
                                     optimal_trans=init_opt, greedy=greedy, by=by)

    if resume_model is not None:
        first_step, total_step = fs, fs

    if unique:
        fix = np.random.normal(0, 1, size=[m, m]) if fixed is None else fixed
        T_fix = tf.Variable(fix, dtype='float64', name='T_fix', trainable=False)  # fixed part of transform
        T = tf.concat([T_fix, params['T']], axis=1)
        params.update({'T': T})  # the full transform
    else:
        T = params['T']

    # prepare for data
    X_t = tf.compat.v1.placeholder(tf.float64, shape=(d, batch_size_phase), name="X")
    Y_t = tf.compat.v1.placeholder(tf.float64, shape=(p, batch_size_phase), name="Y")

    X_v = tf.Variable(np.transpose(x_v), name="Xv", trainable=False, dtype='float64')
    Y_v = tf.Variable(y_v, name="Yv", trainable=False, dtype='float64')

    # forward pass
    if gaussian_assumption:

        if true_cov:
            T_, _ = expand_trans(T, p)
            cov = tf.Variable(Sigma, dtype='float64', trainable=False)
            cov = tf.matmul(tf.matmul(T_, cov), tf.transpose(T_))  # sent to lower dimensional space

            chol = tf.linalg.cholesky(cov[p:, p:])
            loss_t = tf.linalg.det(cov[:p, :p] - tf.matmul(cov[:p, p:],
                                                    tf.linalg.cholesky_solve(chol, cov[p:, :p])))
            loss_v = loss_t

        else:  # actually estimate the covariance
            loss_t = loss_gauss(T, X_t, Y_t)
            loss_v = loss_gauss(T, X_v, Y_v)

    else:  # non-parametric estimation
        if est_type == 'knn':
            ratio_t = density_ratio_knn(T, X_t, Y_t, k_yz_phase, k_z_phase)
            ratio_v = density_ratio_knn(T, X_v, Y_v, k_yz_phase, k_z_phase)

        if est_type == 'kde':
            ratio_t = density_ratio_kde(T, X_t, Y_t, h_phase)
            ratio_v = density_ratio_kde(T, X_v, Y_v, h_phase)

        if est_type == 'kde_gen':  # do not use with diagonal covariance; much slower than kde
            ratio_t = density_ratio_kde_gen(T, X_t, Y_t, A_inv)
            ratio_v = density_ratio_kde_gen(T, X_v, Y_v, A_inv)

        # define loss
        loss_t = tf.reduce_mean(-tf.math.log(ratio_t))  # training loss
        loss_v = tf.reduce_mean(-tf.math.log(ratio_v))  # validation loss

    # define optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_phase/num_batches_to_accum_phase)

    # setup for sub-minibatching
    t_vars = tf.compat.v1.trainable_variables()
    accum_grads = tf.Variable(tf.zeros_like(t_vars[0].initialized_value()), trainable=False)  # container for grads
    zero_op = accum_grads.assign(tf.zeros_like(accum_grads))  # create an operation to set accumulated grads to 0
    batch_grad = optimizer.compute_gradients(loss_t, t_vars)  # compute the gradients w.r.t 'T'
    accum_op = accum_grads.assign_add(batch_grad[0][0])  # create an operation to accumulate the gradients
    train_step_op = optimizer.apply_gradients([(accum_grads, batch_grad[0][1])])  # operation to apply accum_grads

    # define initializer
    init = tf.compat.v1.global_variables_initializer()

    # start training session
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        # initial state of learning assuming normality
        params_val = sess.run(params)
        var_init = analytic_transformation_variance(params_val['T'], Sigma, p)
        print('\n')
        print("Variance of y | T(x) for initialized T: {}".format(var_init))
        print('\n')

        # core training iterations
        for step in range(num_steps_phase):
            sess.run(zero_op)  # reset the accum_grads variable to 0 before starting each step
            for batch in range(num_batches_to_accum_phase):
                batch_x, batch_y = get_minibatch(int(batch_size_phase), x_t, y_t, normalize)
                _, loss_t_val, loss_v_val, params_val = sess.run([accum_op, loss_t, loss_v, params],
                                                     feed_dict={X_t: batch_x, Y_t: batch_y})
            sess.run([train_step_op])  # update 'T' by one training step

            # training diagnostics/documentation
            losses_t.append(loss_t_val)
            losses_v.append(loss_v_val)

            print("Iteration: {};  Training loss: {};  Validation Loss: {}".format(step, np.around(loss_t_val, 6),
                                                                                   np.around(loss_v_val, 6)))
            if synthetic_data:
                if total_step % 10 == 0 and total_step != 0:
                    var_trans = analytic_transformation_variance(params_val['T'], Sigma, p)
                    vars_.append(var_trans)
                    print_variances(True, var_trans, var_opt)

            if save_run:
                np.save(save_path + '/steps.npy', np.array([total_step]))
                np.save(save_path + '/T{}'.format(total_step), params_val['T'])

            total_step += 1

##################################### POST TRAINING ###################################################

# do some plotting of the training
if synthetic_data:
    loss_plot(losses_t, losses_v,
              save_path, first_step, total_step, training_params, data_params)
else:
    loss_plot(losses_t, losses_v,
              save_path, first_step, total_step, training_params, read_data_params)

# save the losses in case we pick up training
np.save(save_path + '/training_losses.npy', losses_t)
np.save(save_path + '/validation_losses.npy', losses_v)
np.save(save_path + '/vars.npy', np.array(vars_))

# clean up the models we no longer need, and do some further saving
if save_run:
    T = clean_up(losses_v, save_path, first_step, total_step)
    if synthetic_data:
        var_trans = analytic_transformation_variance(params_val['T'], Sigma, p)
        with open(save_path + "/variance_report.txt", "w") as file_:
            file_.write("Var(y|T(x)) for T with lowest validation error: {}".format(var_trans))
            file_.write("\n")
            file_.write("Var(y|T^*(x)), i.e. variance under sufficient dim reduction: {}".format(var_opt))
else:
    shutil.rmtree(save_path)

print('\n')
print('Training complete. Results can be found in {}'.format(save_path))
print('\n')