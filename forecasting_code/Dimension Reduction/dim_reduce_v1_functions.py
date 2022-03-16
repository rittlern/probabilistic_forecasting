########################### FILE FOR FUNCTIONS RELATING TO TF 1.x DIM REDUCTION ######################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import scipy.spatial as ss
from scipy.special import digamma
from math import pi as pi
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal as mvn
from  scipy.spatial.distance import cdist


##################################### FOR DATA READING/GENERATION  ###################################


def readdata(files, data, variables_to_get, timerange, dir=None):
    """
    Read data from .netCDF files; from Carlo's experiments on KC grids

    :param files: python list; contains strings of files in a given directory to read in
    :param data: python dictionary; empty in current implementation
    :param variables_to_get: python list; of strings representing variables
    :param timerange: python list; of 2 ints delimiting the timesteps to pull from
    :param dir: string; directory where data stored
    :return: python dictionary; wherein each key-value pair corresponds to variable and it's timeseries
    """
    files.sort()
    for file in files:
        if dir is not None:
            file = dir + "/" + file
        d = nc.Dataset(file)
        vars = list(d.variables.keys())
        for var in vars:
            if var in variables_to_get:
                if var in data:
                    data[var].append(d[var][timerange[0]:timerange[1]])
                else:
                    data[var] = [d[var][timerange[0]:timerange[1]]]


def visualize_training_data(x_train, y_train, path, max_x_to_plot):
    """
    Constructs correlation plots relating a subset of the variables in question; for getting a feel for
    the data on which training occurs

    :param x_train: np.ndarray; training data for predictors; d columns for the d predictors in higher dim space
    :param y_train: np.ndarray; training responses; p number of rows.
    :param path: string; directory where image should be saved
    :param max_x_to_plot: int; how many predictors to include in plots
    :return: None
    """
    from seaborn import pairplot
    from pandas import DataFrame

    max_x_to_plot = x_train.shape[1] if max_x_to_plot > x_train.shape[1] else max_x_to_plot
    yd = y_train.shape[0]

    frame = DataFrame(np.concatenate((np.transpose(y_train), x_train[:, :max_x_to_plot+1]), axis=1))
    if yd == 1:
        col_names = ['Y'] + ['X' + str(i - 1) for i in range(frame.shape[1]-yd+1) if i != 0]
    else:
        col_names = ['Y1'] + ['Y2'] + ['X' + str(i - 1) for i in range(frame.shape[1]-yd+1) if i != 0]
    frame.columns = col_names

    g = pairplot(frame)
    g.fig.suptitle("Partial Correlogram, Training Data", y=1.03)
    g.savefig(path + '/correlogram.png')

    return None


def gp_data_reorder(x, y):
    """
    transforms separate x, y into a data arrangement that is suitable for Gaussian process computations
    :param x: np.ndarray; predictors; d columns for the d predictors in higher dim space
    :param y: np.ndarray; training responses; p number of rows.
    :return: checkered arrangement of predictors where 'up', 'vp' are consecutive for the same location
    """

    d = x.shape[1]
    v = np.concatenate((y, x), axis=1)  # v = y;x
    temp = np.zeros(np.shape(v))  # re-ordering. hold things in place for now

    temp[:, range(2, (d + 1), 2)] = v[:, 2:(2 + int(d / 2))]
    temp[:, range(3, d + 2, 2)] = v[:, (2 + int(d / 2)):]
    temp[:, :2] = y

    return temp


def gp_data_reorder_permutation(d):
    """
    constructs inverse permutation that undoes the data reordering operation for gp computations
    :param d: number of predictors
    :return: permutation matrix
    """

    PI = np.zeros([d+2, d+2])
    PI[:2, :2] = np.eye(2)
    count = 2
    for i in range(2, int(d / 2) + 2):
        PI[i, count] = 1
        count += 2

    count = 3
    for i in range(int(d / 2) + 2, d + 2):
        PI[i, count] = 1
        count += 2

    return PI


################################### FOR TRAINING SETUP ######################################


def path_finder(stem, data_directory):
    """
    Finds the loading and saving paths for the training and returns them.

    :param stem: string; directory where subdirectories corresponding to individual training runs live
    :param data_directory: string/None; subdirectory where the training/validation data are stored for this run.
           can be None, in which case the most recent subdirectory containing data is returned
    :return: tuple of strings; (directory where data are stored, directory where run stats will be stored)
    """

    if data_directory is not None:  # if the user supplies a specific Experiments/run_ to use data from
        path = data_directory
    else:
        all_subdirs = [stem + f + '/' for f in os.listdir(stem) if os.path.isdir(stem + f)]
        valid_subdirs = [direc for direc in all_subdirs if 'x_train.npy' in os.listdir(direc)]
        assert len(valid_subdirs) != 0, 'There are no subdirectories containing data. Run a generation/reading script'
        path = max(valid_subdirs, key=os.path.getmtime)  # take the data used in the latest saved run

    empty = len([f for f in os.listdir(path) if f[0] == 'T']) == 0  # are there transformations saved here?

    try:
        largest_run_number = max([int(sub.split('/')[1].split('_')[1]) for sub in all_subdirs])
    except ValueError:
        raise Exception("Be sure to remove all runs except those ending in integers before training.")
    save_path = path.split('/')[0] + '/' + 'run_' + str(largest_run_number + 1) + '/' if not empty else path

    return path, save_path


def save_settings(path, save_path, data_params, training_params, synthetic_data):
    """
    Save the training and data settings to the directory where training results stored

    :param path: string; directory path where training data are stored
    :param save_path: string; directory path where training results stored
    :param data_params: python dictionary; either generation or read settings dictionary
    :param training_params: python dictionary; training settings dictionary
    :param synthetic_data: boolean, whether data are synthetic
    :return: None
    """

    if save_path != path:
        os.mkdir(save_path)
        ofd = open(save_path + 'data_settings.txt', "w")
        if synthetic_data:
            ofd.write('##########################\n#\n# Data Generation Parameters:\n')
        else:
            ofd.write('##########################\n#\n# Data Reading Parameters:\n')
        for l in data_params.items():
            ofd.write("#J# " + l[0] + " = " + repr(l[1]) + "\n")
        ofd.close()

    ofd = open(save_path + 'training_settings.txt', "w")
    ofd.write('##########################\n#\n# Training Hyperparameters:\n')
    for l in training_params.items():
        ofd.write("#J# " + l[0] + " = " + repr(l[1]) + "\n")
    ofd.close()

    return None


def load_gen_params(path, generation_type, p):
    """
    Loads mean and covariance matrix used to generate data if data are synthetic.

    :param path: string; directory path where the data are stored
    :param generation_type: int; number of data generation process
    :param p: int; dimension of response
    :return: tuple of np.ndarrays; (mean, covariance)"""

    if generation_type != 1 and generation_type != 2:
        if p == 2:
            gamma_1, gamma_2 = np.load(path + 'gamma_1.npy'), np.load(path + 'gamma_2.npy')
        else:
            gamma_1, gamma_2 = np.load(path + 'gamma.npy'), None
        Sigma_xx = np.load(path + 'cov.npy')
        Sigma = covariance_structure(p, Sigma_xx, 1, gamma_1, gamma_2)
        mu = None
    if generation_type == 1 or generation_type == 2:
        Sigma = np.load(path + 'cov.npy')
        mu = np.load(path + 'mu.npy')

    return mu, Sigma


def lat_long_t(ind):
    """
    compute the spatio-temporal coordinates for each data point in accordance with the gp_data_reorder ordering
    this is currently hard-coded for 19x19 grid and 2d response
    :param ind: index in the temp matrix as a list
    :return: coordinate in space (lat, long, t)
    """
    coord = [0, 0, 0]
    if ind[1] == 0 or ind[1] == 1:  # we're dealing with response
        coord = [9, 9, 1]  # magnitude of this time coordinate is important in the training if time/space together
    else:  # we're dealing with predictors
        # ind[1]-2 is there as first 2 columns are y_t
        coord[0], coord[1] = np.floor((ind[1]-2) / 38), np.floor((ind[1]-2)/2) % 19  # 18 is just python indexing;  18- ((ind[1]-2)/2) % 19before

    # if ind[1] % 2 == 1:
    #     coord[0], coord[1] = coord[1], coord[0]  # need transpose for 'up'
    return np.array(coord)


####################################### FOR TRAINING #######################################


def get_minibatch(batch_size, x_train, y_train, normalize=False):
    """
    Draws 'batch_size' samples from empirical CDF of the training data.

    :param batch_size: int; number of samples to draw
    :param x_train: np.ndarray; N x d array of predictor samples
    :param y_train: np.ndarray; p x N array of response samples
    :param normalize: boolean; if True, normalize the batch
    :return: tuple of np.ndarray; (d x batch_size predictors batch, p x batch_size response batch)
    """

    nn = x_train.shape[0]
    idx = np.random.choice(np.array(list(range(0, nn))), size=batch_size, replace=False)
    batch_x = x_train[idx, :]
    batch_y = y_train[:, idx]

    if normalize:
        batch_x = (batch_x - np.mean(batch_x, axis=0)) / np.std(batch_x, axis=0)
        batch_y = np.transpose((np.transpose(batch_y) - np.mean(batch_y, axis=1)) / np.std(batch_y, axis=1))

    return np.transpose(batch_x), batch_y


def set_T(phase, training_phases, m, d, yd, training_seed, unique=False,
          resume_model=None, resume_model_dir=None, optimal_trans=None, greedy=False, by=1):
    """
    Initialize, or if continuing training, load specified transformation of predictors to lower dimensional
    space.

    :param phase: int; phase of the training
    :param training_phases: python list; of integers specifying all the training phases
    :param m: int; reduced dimension of the predictors
    :param d: int; original dimension of the predictor space
    :param yd: int; dimension of the response
    :param training_seed: int; seed used for initialize of any random components
    :param unique: boolean; if True, we fix part of the transformation to eliminate gauge freedom
    :param resume_model: string/None; name of .npy file where the model to resume on is stored; if None, start over
    :param resume_model_dir: string/None; directory where 'resume_model' lives
    :param optimal_trans: np.ndarray; optimal transformation (presumably under normal theory) for non-random init
    :param greedy: boolean; ignore unless running 'reduction_optimality.py'; if True, will in part use optimal model
           from training session on a smaller m to initialize the new training session
    :param by: int; ignore unless running 'reduction_optimality.py'; specify the gap between m's to train on
    :return: tuple; (dictionary containing trainable tensor, current iteration of training, )
    """

    first_step = 0
    fixed = None
    higher_dim = d - m if unique else d

    assert not (optimal_trans is not None and resume_model), \
        'Cannot resume training model and initialize with optimal normal T.'
    assert not (optimal_trans is not None and unique), \
        'No support for unique transformation when providing optimal trans as of now.'
    assert not (greedy and unique), 'No support for unique transformation and greedy updates as of now.'

    if phase == training_phases[0]:
        if optimal_trans is not None:
            # init = tf.Variable(optimal_trans, dtype='float64', trainable=True, name='opt_normal')
            # init_rest = tf.Variable(tf.random.normal([ m - yd, d], mean=0, stddev=.01, dtype='float64',
            #                                          seed=training_seed[0]), name="linear_operator", trainable=True)
            # T = tf.concat([init, init_rest], axis=0)

            init1 = tf.Variable(optimal_trans[:yd, :yd], dtype='float64', trainable=False, name='init1')
            init2 = tf.Variable(tf.random.normal([yd, d-yd], mean=0, stddev=.1, dtype='float64',
                                                      seed=training_seed[0]), name="linear_operator", trainable=True)
            init_rest = tf.Variable(tf.random.normal([m - yd, d], mean=0, stddev=.1, dtype='float64',
                                                      seed=training_seed[0]), name="linear_operator", trainable=True)
            T = tf.concat([init1, init2], axis=1)
            T = tf.concat([T, init_rest], axis=0)

        elif resume_model is not None:
            try:
                t = np.load(resume_model_dir + resume_model)

                if unique:
                    fixed = t[:, :m]
                    t = t[:, m:]
                    assert (higher_dim == t.shape[1]), 'Trainable part of unique transformation must be (m) x (m-d).'

                if greedy:  # we're loading a smaller model, so we need to concat an extra row
                    extra_rows = np.random.normal(0, .01, size=[by, higher_dim])
                    t = np.concatenate((t, extra_rows), axis=0)
                assert m == t.shape[0], 'Transformations must be m x d.'

                trainable_params = {'T': tf.Variable(t, name="linear_operator", trainable=True)}
                first_step = int(np.load(resume_model_dir + 'steps.npy'))
                print("Resuming training...")

            except (OSError, TypeError):
                print("Couldn't find the model to resume on in specified directory. Using random initialization...")
                T = tf.Variable(tf.random.normal([m, higher_dim], mean=0, stddev=.01, dtype='float64',
                                                 seed=training_seed[0]), name="linear_operator", trainable=True)
                trainable_params = {'T': T}

        else:
            T = tf.Variable(tf.random.normal([m, higher_dim], mean=0, stddev=.01, dtype='float64',
                                             seed=training_seed[0]), name="linear_operator", trainable=True)
        if resume_model is None:
            trainable_params = {'T': T}

    else:
        # load most recent from path; this is where the last model of the previous phase is stored
        all_files_in_path = [path + f for f in os.listdir(path)]
        t_file = max(all_files_in_path, key=os.path.getctime)
        t = np.load(t_file)  # most recent learned transformation in path
        trainable_params = {'T': tf.Variable(t, name="linear_operator", trainable=True)}

    return trainable_params, fixed, first_step


def loss_gauss(T, x, y):
    """
    Computes estimate for det(Sig_11 - Sig_12 * Sig_22^(-1) * Sig_21), i.e. generalized variance y|T(x).

    :param T: tf.Variable; tensor version of the transformation of predictors
    :param x: tf.Tensor/tf.Variable; placeholder tensor for x training or variable containing validation examples
    :param y: tf.Tensor/tf.Variable; placeholder tensor for y training or variable containing validation examples
    :return: tf.Tensor; 1x1 tensor with the loss of the procedure
    """

    yd = y.shape[0]  # in tf 1.x, y.shape[0].value
    m = T.shape[0]  # in tf 1.x, T.shape[0].value
    batch_size = tf.constant(x.shape[1], dtype='float64')  # in tf 1.x, x.shape[1].value

    Z = tf.linalg.matmul(T, x)
    Z = tf.concat((y, Z), axis=0)

    mu = tf.reshape(tf.reduce_mean(Z, axis=1), [(m + yd), 1])
    centered = tf.transpose(Z - mu)
    cov = tf.reduce_sum(tf.einsum('ij...,i...->ij...', centered, centered), axis=0) / batch_size  # in lower dim space

    chol = tf.linalg.cholesky(cov[yd:, yd:])
    est = tf.linalg.det(cov[:yd, :yd] - tf.matmul(cov[:yd, yd:],
                                                      tf.linalg.cholesky_solve(chol, cov[yd:, :yd])))
    return est


def density_ratio_knn(T, x, y, k_yz, k_z):
    """
    Computes estimate for ratio of densities ( i.e. p(y|z) ) at each of the sample points
    using k-nearest neighbors.

    :param T: tf.Variable; tensor version of the transformation of predictors
    :param x: tf.Tensor/tf.Variable; placeholder tensor for x training or variable containing validation examples
    :param y: tf.Tensor/tf.Variable; placeholder tensor for y training or variable containing validation examples
    :param k_yz: int; k for the knn density estimation of p(y,z)
    :param k_z: int; k for the knn density estimation of p(z)
    :return: tf.Tensor; batch_size-length tensor containing est for p(y_i | z_i)
    """

    p = y.shape[0]  # y.shape[0].value in tf 1.x
    m = T.shape[0]
    batch_size = x.shape[1]

    Z = tf.linalg.matmul(T, x)
    yZ = tf.transpose(tf.concat((y, Z), axis=0))

    V = tf.reshape(yZ, (batch_size, 1, p + m)) - tf.reshape(yZ, (1, batch_size, p + m))
    Vz = tf.reduce_prod(V[:, :, p:], axis=2)
    Vy = tf.reduce_prod(V[:, :, :p], axis=2)
    Vyz = Vz * Vy

    vol_z = -tf.math.top_k(-tf.abs(Vz), k_z+1)[0][:, k_z]  # flexible to choice of k nearest neighbors
    vol_yz = -tf.math.top_k(-tf.abs(Vyz), k_yz+1)[0][:, k_yz]
    est = (vol_z / vol_yz) * (k_yz / k_z)

    return est


def density_ratio_kde(T, x, y, lambda_):
    """
    Computes density ratios estimate at data points via kde with smoothing parameter lambda_.
    Uses gaussian kernels with no correlation. If looking to compare this to some theoretical baseline,
    it's vital that correct constants used in the estimation.

    :param T: tf.Variable; tensor version of the transformation of predictors
    :param x: tf.Tensor/tf.Variable; placeholder tensor for x training or variable containing validation examples
    :param y: tf.Tensor/tf.Variable; placeholder tensor for y training or variable containing validation examples
    :param lambda_: np.ndarray; m+p length array of smoothing values
    :return: tf.Tensor; batch_size-length tensor containing est for p(y_i | z_i)
    """

    p = y.shape[0]  # y.shape[0].value in tf 1.x
    m = T.shape[0]
    batch_size = x.shape[1]

    Zx = tf.linalg.matmul(T, x)
    Zy = tf.transpose(tf.concat((y, Zx), axis=0))

    Z = tf.reshape(Zy, (batch_size, 1, m+p)) - tf.reshape(Zy, (1, batch_size, m+p))
    Z2H = tf.math.square(Z) / tf.broadcast_to(tf.constant(lambda_, dtype='float64'), [batch_size, batch_size, m+p])

    norms_z = tf.math.reduce_sum(Z2H[:, :, p:], axis=2)
    norms_y = tf.math.reduce_sum(Z2H[:, :, :p], axis=2)
    norms_yz = norms_z + norms_y  # marginally faster than 2 calls to tf.norm

    kde_z_ = tf.reduce_sum(tf.math.exp(-norms_z / 2), axis=0)
    kde_yz_ = tf.reduce_sum(tf.math.exp(-norms_yz / 2), axis=0)
    est = kde_yz_ * ((2 * pi)**(-p/2)) * (np.prod(lambda_[:p])**(-.5)) / kde_z_

    return est


def density_ratio_kde_gen(T, x, y, A_inv):
    """
    Computes density ratios estimate at data points via kde via gaussian kernels with correlation
    A. Should not be used with a diagonal A as this function is much less efficient in that specific
    regime than 'density_ration_kde'. Only computes density up to a constant at this point.

    One option for A would be the empirical covariance in the lower dimensional space.

    :param T: tf.Variable; tensor version of the transformation of predictors
    :param x: tf.Tensor/tf.Variable; placeholder tensor for x training or variable containing validation examples
    :param y: tf.Tensor/tf.Variable; placeholder tensor for y training or variable containing validation examples
    :param A: np.ndarray; precision matrix defining the kernels
    :return: tf.Tensor; batch_size-length tensor containing est for p(y_i | z_i)
    """

    p = y.shape[0].value
    m = T.shape[0].value
    batch_size = x.shape[1].value

    Zx = tf.linalg.matmul(T, x)
    Zy = tf.transpose(tf.concat((y, Zx), axis=0))

    A_inv = tf.constant(A_inv)
    Z = tf.reshape(Zy, (batch_size, 1, m + p)) - tf.reshape(Zy, (1, batch_size, m + p))
    AZ = tf.einsum('lk...,ijk...->ijk...', A_inv, Z)

    ZAZ = Z * AZ

    norms_z = tf.math.reduce_sum(ZAZ[:, :, p:], axis=2)
    norms_y = tf.math.reduce_sum(ZAZ[:, :, :p], axis=2)
    norms_yz = tf.math.add(norms_z, norms_y)   # marginally faster than 2 calls to tf.norm

    kde_z_ = tf.reduce_sum(tf.exp(-norms_z / 2), axis=0)
    kde_yz_ = tf.reduce_sum(tf.exp(-norms_yz / 2), axis=0)
    est = kde_yz_ / kde_z_

    return est


def loss_baseline_gaussian(mu, Sig, p):
    """
    Analytically computes loss E_(y,x) -log(p(y,T^*(x))/p(T(x))) under sufficient dim reduction transformation T
    and true density. For evaluating runs on synthetic data (y,x) ~ N(mu, Sig). This shouldn't be used
    unless bias correction is done in the training loss.

    :param mu: np.ndarray; true mean used for data generation of (y,x)
    :param Sig: np.ndarray; true covariance used for data generation of (y,x)
    :param p: int; dimension of response
    :return: float; E_(y,x) -log(p(y,T^*(x))/p(T(x)))
    """

    T_opt = np.transpose(np.linalg.solve(Sig[p:, p:], Sig[p:, :p]))
    d = Sig.shape[0]  # here Sig has size of original dimension of x's + p

    T_yz, T_z = expand_trans_npy(T_opt, p)

    Sig_yz = np.matmul(T_yz, np.matmul(Sig, np.transpose(T_yz)))
    Sig_z = np.matmul(T_z, np.matmul(Sig, np.transpose(T_z)))
    A_yz = np.matmul(np.transpose(T_yz), np.linalg.solve(Sig_yz, T_yz))
    A_z = np.matmul(np.transpose(T_z), np.linalg.solve(Sig_z, T_z))

    Mu = np.matmul(np.reshape(mu, (d, 1)), np.transpose(np.reshape(mu, (d, 1))))
    mu_yz = np.matmul(T_yz, mu)
    mu_z = np.matmul(T_z, mu)

    baseline = .5 * ( p * np.log(2 * pi) + np.linalg.slogdet(Sig_yz)[1] - np.linalg.slogdet(Sig_z)[1] +
                     np.trace(np.matmul(A_yz - A_z, Mu + Sig)) -
                     np.matmul(mu_yz, np.linalg.solve(Sig_yz, mu_yz)) +
                     np.matmul(mu_z, np.linalg.solve(Sig_z, mu_z)) )

    return baseline


def loss_baseline_gaussian_kde_est(Sig, H, p):
    """
    Estimate the baseline (i.e. loss under the sufficient dim reduction transformation) for the loss, including bias,
    when using kde with gaussian kernels given by matrix H.

    :param mu: np.ndarray; mean used to generate the data
    :param Sig: np.ndarray; covariance used to generate the data
    :param p: int; dimension of the response
    :param H: np.ndarray; covariance matrix used in definition of kernels
    :return: float; est
    """

    T_opt = np.transpose(np.linalg.solve(Sig[p:, p:], Sig[p:, :p]))
    T_yz, T_z = expand_trans_npy(T_opt, p)

    Sig_yz_h = np.matmul(np.matmul(T_yz, Sig), np.transpose(T_yz)) + H
    Sig_z_h = np.matmul(np.matmul(T_z, Sig), np.transpose(T_z)) + H[p:, p:]

    A_yz = np.linalg.inv(Sig) + np.matmul(np.transpose(T_yz), np.linalg.solve(Sig_yz_h, T_yz))
    A_z = np.linalg.inv(Sig) + np.matmul(np.transpose(T_z), np.linalg.solve(Sig_z_h, T_z))

    est = .5 * ( np.linalg.slogdet(Sig_yz_h)[1] + np.linalg.slogdet(A_yz)[1] -
                np.linalg.slogdet(Sig_z_h)[1] - np.linalg.slogdet(A_z)[1] +
                    p * np.log(2*pi) )
    return est


def loss_baseline_gaussian_kde_mc(mu, Sig, H, p, normalize, n=1, batch_size=3000):
    """
    Computes an estimate of the loss baseline via Monte Carlo integration. Assumes that diagonal kernels
    used for speed of computation but this could easily be amended. This function should be highly preferred
    over loss_baseline_gaussian_kde for accurate estimates.

    :param mu: np.ndarray; mean used to generate the data
    :param Sig: np.ndarray; covariance used to generate the data
    :param H: np.ndarray; covariance matrix used in definition of kernels
    :param p: int; dimension of the response
    :param normalize: bool; must specify if batches are being normalized to get the same estimate
    :param n: int; number of simulations to run
    :param batch_size: int; size of each simulation
    :return: float; est
    """

    T_opt = np.transpose(np.linalg.solve(Sig[p:, p:], Sig[p:, :p]))
    T_yz, _ = expand_trans_npy(T_opt, p)
    m = T_opt.shape[0]

    # simulate the data
    mc_data = np.empty([batch_size, m+p, n])
    for i in range(n):
        data = np.random.multivariate_normal(mu, Sig, size=batch_size)
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) if normalize else data
        mc_data[:, :, i] = np.transpose(np.matmul(T_yz, np.transpose(data)))

    # run experiments
    estimates = np.empty(n)
    for i in range(n):

        diffs = np.reshape(mc_data[:, :, i], (batch_size, 1, m+p)) - \
                np.reshape(mc_data[:, :, i], (1, batch_size, m+p))
        m_diffs = diffs**2 / np.diagonal(H)
        norms_z = np.sum(m_diffs[:, :, p:], axis=2)
        norms_y = np.sum(m_diffs[:, :, :p], axis=2)
        norms_yz = norms_z + norms_y

        kde_z = np.sum(np.exp(-norms_z / 2), axis=0)
        kde_yz = np.sum(np.exp(-norms_yz / 2), axis=0)
        conditional_est = kde_yz * (2*pi)**(-p/2) * np.prod(np.diagonal(H)[:p])**(-.5) / kde_z
        estimates[i] = np.mean(-np.log(conditional_est))

    return np.mean(estimates)


def loss_baseline_gaussian_knn_mc(mu, Sig, k_yz, k_z, normalize, p, n=1, batch_size=700):
    """
    Exactly the same as 'loss_baseline_gaussian_kde_mc' but for knn estimation

    :param mu: np.ndarray; mean used to generate the data
    :param Sig: np.ndarray; covariance used to generate the data
    :param k_yz: int; k used in estimation in yz space
    :param k_z: int; k used in estimation in z space
    :param normalize: bool; must specify if batches are being normalized to get the same estimate
    :param p: int; dimension of the response
    :param n: int; number of simulations to run
    :param batch_size: int; size of each simulation
    :return: float; est
    """

    T_opt = np.transpose(np.linalg.solve(Sig[p:, p:], Sig[p:, :p]))
    T_yz, _ = expand_trans_npy(T_opt, p)
    m = T_opt.shape[0]

    # simulate the data
    mc_data = np.empty([batch_size, m + p, n])
    for i in range(n):
        data = np.random.multivariate_normal(mu, Sig, size=batch_size)
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) if normalize else data
        # normalize seems like it could be a problem, but this happens after anything parametric happens
        mc_data[:, :, i] = np.transpose(np.matmul(T_yz, np.transpose(data)))

    # run experiments
    estimates = np.empty(n)
    for i in range(n):
        diffs = np.reshape(mc_data[:, :, i], (batch_size, 1, p + m)) - \
                np.reshape(mc_data[:, :, i], (1, batch_size, p + m))
        Vz = np.prod(diffs[:, :, p:], axis=2)
        Vy = np.prod(diffs[:, :, :p], axis=2)
        Vyz = Vy * Vz

        vol_z = np.partition(np.abs(Vz), k_z, axis=1)[:, k_z]
        vol_yz = np.partition(np.abs(Vyz), k_yz, axis=1)[:, k_yz]

        estimates[i] = np.nanmean( -np.log( (vol_z / vol_yz) * (k_yz / k_z) ) )
        print(estimates[i])

    return np.mean(estimates)


def expand_trans(T, p):
    """
    Creates tensorflow compatible transformations that are augmented to preserve y data when multiplied by (y,x),
    and remove the y data, respectively, i.e. creates T_{yz} given T.

    :param T: tf.Variable; tensor version of the transformation of predictors
    :param p: int; dimension of response
    :return: tf.Tensor; augmented transformations
    """

    m, d = T.shape[0], T.shape[1]
    top = np.zeros([p, d+p])
    top[0, 0] = 1
    if p == 2:
        top[1, 1] = 1
    top = tf.constant(top, dtype="float64")

    left = np.zeros([m, p])
    left = tf.constant(left, dtype="float64")

    T_z = tf.concat([left, T], axis=1)  # can use paddings for this too
    T_yz = tf.concat([top, T_z], axis=0)

    return T_yz, T_z


def expand_trans_npy(T, p):
    """
    Same functionality as expand_trans but for numpy arrays.

    :param T: np.ndarray; transformation of predictors
    :param p: int; dimension of response
    :return: np.ndarray; augmented transformations
    """

    m, d = T.shape[0], T.shape[1]

    T_yz = np.zeros([m + p, d + p])
    upper_left = np.diag(np.full(p, 1))
    T_yz[:p, :p] = upper_left
    T_yz[p:, p:] = T

    T_z = np.zeros([m, d + p])
    T_z[:, p:] = T

    return T_yz, T_z


def neg_likelihood_gp(theta, Y, differences, time_mat, noise):
    """negative likelihood for GP given data matrix Y up to additive constant. minimize this objective"""

    d = differences.shape[0]
    box_ = np.eye(2)  # learning E/W and N/S independently
    tiled = np.tile(box_, (int(d/2), int(d/2)))
    nugget = noise * np.eye(d)

    f = np.array([theta[1], theta[2]])
    F = np.repeat(f, d**2).reshape([d, d, 2], order='F')
    T = np.repeat(time_mat[:, :, np.newaxis], 2, axis=2)

    K_tilde = np.exp(- (np.linalg.norm(differences + F * T, ord=2, axis=2)**2) * theta[0]**2) * tiled + nugget
    chol = cho_factor(K_tilde)
    logdet = np.sum(np.log(np.diag(chol[0]))) * 2  # log rules and properties of determinant yield this

    print('Inverse characteristic length scale: {}'.format(abs(theta[0])))

    return Y.shape[1] * np.log(np.trace(np.matmul(Y, cho_solve(chol, np.transpose(Y))))) \
           + logdet

##################################### FOR TRAINING EVALUATION #######################################


def mutual_information_est(z, y, k):
    """
    Non-parametrically estimates the mutual information of z and y using the estimation
    method of Kraskov (2003). Fast implementation using cython kdtrees due to Greg Ver Steeg's entropy package.

    The following serves as a nice test of the function:
        # rho = .5
        # x = np.random.multivariate_normal(np.array([0, 0]), np.array([[1,rho], [rho,1]]), size = 1000)
        # z = x[:, 1]
        # y = x[:, 0]
        # print(-.5*np.log(1-rho**2))
        # print(mutual_information_est(z,y,10))

    :param z: np.ndarray; N x d1 array of samples
    :param y: np.ndarray; N x d2 array of samples
    :param k: int; k for nearest neighbor searches
    :return: float; mutual information estimate
    """

    def avgdigamma(points, d_vec):
        """
        Helper function for mutual_information_est. This part finds number of neighbors in some radius in
        the marginal space, and returns expectation value of <psi(nx)> from p.2 of Kraskov (2003)

        :param points: np.ndarray; N x di array of samples from one of the variables
        :param d_vec: np.ndarray; vector of distances is the full (y,z) space
        :return: float; <psi(n_x/y)>
        """

        n_elements = len(points)
        tree = ss.cKDTree(points.reshape([n_elements, 1]))
        avg = 0.
        d_vec = d_vec - 1e-15  # for stability
        for point, dist in zip(points, d_vec):
            num_points = len(tree.query_ball_point([point], dist, p=float('inf')))
            avg += digamma(num_points) / n_elements
        return avg

    points = [z, y]
    points = np.transpose(np.vstack(points))
    tree = ss.cKDTree(points)
    d_vec = tree.query(points, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]

    return digamma(k) + digamma(len(z)) - avgdigamma(z, d_vec) - avgdigamma(y, d_vec)


def covariance_structure(p, cov, sig_2, beta1, beta2=None):
    """
    Analytically computes covariance matrix of (y,x) for the y_i|x ~ N(beta_i^Tx, sig^2), x ~ N(mu, Sigma)
    data generation processes using the true parameters. Responses always independent given x with conditional
    var = sig_2.

    :param p: float; dimension of response
    :param cov: np.ndarray; covariance matrix governing the predictors
    :param sig_2: float; varianace of the response variables
    :param beta1: np.ndarray; linear transformation of predictors to give mean of y_1
    :param beta2: np.ndarry; linear transformation of predictors to give mean of y_2
    :return: np.ndarray; expanded covariance matrix
    """

    n = len(beta1) + p
    Sigma = np.zeros(shape=[n, n])
    Sigma[0, 0] = np.matmul(np.matmul(np.transpose(beta1), cov), beta1) + sig_2

    if p == 1:
        Sigma[0, p:] = np.matmul(np.transpose(beta1), cov)
        Sigma[p:, 0] = np.transpose(Sigma[0, p:])

    if p == 2:
        Sigma[1, 1] = np.matmul(np.matmul(np.transpose(beta2), cov), beta2) + sig_2
        Sigma[0, p:] = np.matmul(np.transpose(beta1), cov)
        Sigma[p:, 0] = np.transpose(Sigma[0, p:])
        Sigma[1, p:] = np.matmul(np.transpose(beta2), cov)
        Sigma[p:, 1] = np.transpose(Sigma[1, p:])
        Sigma[0, 1] = np.dot(beta1, np.matmul(cov, beta2))
        Sigma[1, 0] = np.dot(beta1, np.matmul(cov, beta2))

    Sigma[p:, p:] = cov

    return Sigma


def analytic_transformation_variance(T, Sigma, p):
    """
    Computes the variance of y|T(x) given either true or estimated covariance structure of (y,x) in the
    normal case

    :param T: np.ndarray; transformation of predictors to lower dimensional space
    :param Sigma: np.ndarray; covariance matrix of (y,x)
    :param p: float; dimension of response
    :return: float; generalized variance
    """

    T_ = np.zeros(shape=[T.shape[0]+p, T.shape[1]+p])
    T_[p:, p:] = T
    T_[:p, :p] = np.eye(p)
    Gamma = np.matmul(np.matmul(T_, Sigma), np.transpose(T_))

    return np.linalg.det(Gamma[:p, :p] - np.matmul(Gamma[:p, p:],
                                                np.linalg.solve(Gamma[p:, p:], Gamma[p:, :p])))


def print_variances(synthetic_data, var_trans, var_no_reduce=None):
    """
    Handles the complexities of printing variance updates

    :param synthetic_data: boolean; whether or not the data are synthetic
    :param var_trans: float; variance of the transformation T at current state in the training
    :param var_no_reduce: float; variance under sufficient dimension reduction
    :return: None
    """

    print('\n')
    print("Variance of y|T(x), current learned transformation: {}".format(var_trans))
    if synthetic_data:
        print("Variance of y|x, i.e. without dimension reduction: {}".format(var_no_reduce))
    print('\n')
    return None


def clean_up(validation_losses, save_path, first_step, total_step):
    """
    Removes saved transformations that are not interesting and return best transformation.

    :param validation_losses: python list; of validation losses
    :param save_path: string; directory path where model stats have been saved
    :param checkpoints: python list; of training checkpoints
    :return: np.ndarray; transformation with lowest validation loss
    """

    T = None
    best_model_iter = validation_losses.index(min(np.array(validation_losses))) + first_step
    for data_file in os.listdir(save_path):
        if data_file[0] == 'T':
            filename = data_file.split('.')[0]
            if str(best_model_iter) == filename[1:]:
                T = np.load(save_path + '/' + data_file)
            if str(best_model_iter) != filename[1:] and str(total_step) not in filename:
                        os.remove(save_path + '/' + data_file)
    return T


def loss_plot(training_losses, validation_losses, save_path, first_step, total_step, training_dict, data_dict):
    """
    Create training/validation loss plots and save them.

    :param training_losses: python list; of training losses after each training step
    :param validation_losses: python list; of validation losses after each training step
    :param save_path: string; path where training stats saved
    :param first_step: int; first step number of this training session (not 0 if resuming training)
    :param total_step: int; final step of this training session
    :param training_dict: python dictionary; dict containing properties of the training from 'dim_reduce_job.py'
    :param data_dict: python dictionary; dict containing properties of the data from reading/generation script
    :return: None
    """

    steps = list(range(first_step, total_step))
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(steps, training_losses)
    axs[1].plot(steps, validation_losses)
    axs[0].set_title('Training Loss')
    axs[1].set_title('Validation Loss')
    axs[0].set(xlabel='Iteration', ylabel='Loss')
    axs[1].set(xlabel='Iteration', ylabel='Loss')
    plt.figtext(-1, .88, "Original Dimension: {}".format(data_dict['d']), transform=plt.gca().transAxes)
    plt.figtext(-1, .81, "Low Dimension: {}".format(training_dict['m']), transform=plt.gca().transAxes)
    try:
        plt.figtext(.15, .88, "Generation Type: {}".format(data_dict['generation_type']),transform=plt.gca().transAxes)
    except KeyError:
        plt.figtext(.15, .88, "Predictors: {}".format(data_dict['predictors']), transform=plt.gca().transAxes)
        plt.figtext(.15, .81, "Responses: {}".format(data_dict['responses']), transform=plt.gca().transAxes)
    fig.savefig(save_path + '/loss_graph_step{}_to_step{}.png'.format(first_step, total_step))
    plt.clf()

    return None


def trans_plot(T, unique, predictors, sq_size, save_path):
    """
    Implements Carlo's thought of a series of heatmaps showing which coordinates are targeted by the
    learned transformation.

    :param T: np.ndarray; transformation of interest
    :param unique: boolean; is the transformation unique or not?
    :param predictors: python list; of predictors supplied to 'read_data.py' via 'read_data_job.py'
    :param sq_size: int; number of grid points on one side of the square
    :param save_path: string; path where training stats saved
    :return: None
    """

    m = T.shape[0]
    T = np.concatenate((np.eye(m), T), axis=1) if unique else T

    for pred in range(len(predictors)):
        if m % 5 == 0:
            fig, ax = plt.subplots(int(m / 5), 5)
        if m % 4 == 0:
            fig, ax = plt.subplots(int(m / 4), 4)
        if m % 3 == 0:
            fig, ax = plt.subplots(int(m / 3), 3)
        if m % 2 == 0:
            fig, ax = plt.subplots(int(m / 2), 2)

        #fig.suptitle('Transformation Heatmap for {}'.format(predictors[pred]), y=1.08)

        ax = ax.ravel()
        for row in range(m):
            heat_mat = np.empty([sq_size, sq_size])
            count = 0
            for lat in range(sq_size):
                for long in range(sq_size):
                    heat_mat[sq_size - long - 1, lat] = np.abs(T[row, (pred * (sq_size**2)) + count])
                    count += 1
            ax[m*pred + row].set_xticks([])
            ax[m*pred + row].set_yticks([])
            ax[m * pred + row].set_title('Row {}'.format(row+1))
            im = ax[m*pred + row].imshow(heat_mat)
        fig.subplots_adjust(right=0.7)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(save_path + 'transformation_plot.png')
        plt.clf()

    return None

