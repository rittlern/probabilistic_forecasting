import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import scipy.spatial as ss
from scipy.special import digamma
from math import pi as pi
from scipy.stats import gaussian_kde
from scipy.linalg import cho_factor, cho_solve

######################################### DIMREDUCE NETWORK ###################################################


class FullyConnected(tf.Module):
    """
    fully connected layer for dimension reduction
    """

    def __init__(self, name, input_dim, output_dim, initial_mat=None,
                 initial_bias=None, train_bias=False, seed=None):
        """
        constructor

        :param name: string; name of the layer
        :param input_dim: int; dimension of the input vector to this layer
        :param output_dim: int; dimension of the vector after applying this layer
        :param initial_mat: np.ndarray; matrix at which to init linear part of this layer
        :param initial_bias: np.ndarray; vector at which to init non-linear part of layer
        :param train_bias: bool; whether or not to train the bias (False -> learn linear transformation)
        :param seed: int; seed for init
        """
        super(FullyConnected, self).__init__(name=name)

        with self.name_scope:

            self.input_dim, self.output_dim = input_dim, output_dim

            if initial_mat is None:
                initial_mat = tf.random.normal((self.output_dim, self.input_dim), mean=0, stddev=.01, seed=seed)
            else:
                assert input_dim == initial_mat.shape[1]
                assert output_dim == initial_mat.shape[0]

            if initial_bias is None:
                initial_bias = tf.zeros([output_dim, 1])

            self.b = tf.Variable(initial_bias, name="bias", trainable=train_bias)
            self.T = tf.Variable(initial_mat, name='T', trainable=True)

    @tf.Module.with_name_scope
    def forward(self, x):
        """
        compute the affine transformation defined by the layer at a current point in training at input x

        :param x: tf.Variable; vector or matrix whose columns are vectors at which to evaluate transformation
        :return: tf.Variable;
        """
        with tf.name_scope("forward"):
            out = tf.matmul(self.T, x) + self.b
        return out


class PReLU(tf.Module):
    """
    class for PReLU activation function. The function is, element-wise,

           | exp(alpha)*x for x<0
    f(x) = |
           | x for x>0

    where alpha is a vector of dimension equal to that of x.
    """

    ########################################################################
    def __init__(self, name, size=None, initial_alpha=None):
        """
        constructor

        :param name: string; name of the PreLU layer
        :param initial_alpha: np.ndarray; inital value of parameter vector
        """
        super(PReLU, self).__init__(name=name)
        with self.name_scope:
            if initial_alpha is None:
                initial_alpha = -1.0 * tf.reshape(tf.ones(size), (size, 1))

            self.alpha = tf.Variable(initial_alpha, name="alpha", trainable=True, dtype=tf.float32)
            self.slopes_ = tf.math.exp(self.alpha)

    ########################################################################
    @tf.Module.with_name_scope
    def forward(self, x):
        """
        applies PreLU activation to the input x

        :param x: tf.Variable; vector or more commonly, matrix whose columns are vector at which to eval PreLU
        :return: tf.Variable;
        """

        with tf.name_scope("forward"):
            self.slopes_ = tf.math.exp(self.alpha)
            y = tf.where(x > 0, x, x * self.slopes_)
        return y


class DimReduce(tf.Module):
    """
    class for handling dimension reduction transformation learning
    """

    def __init__(self, layers, d, m, p, method, seed, resnet, hyperparams_dict=None, name="DimReduce"):
        """
        constructor

        :param layers: list of tf.Module subclassed layers (e.g. FullyConnected); specify the architecture
        :param d: int; original dimension of the predictor space
        :param m: int; dimension to which we're reducing predictors
        :param p: int; dimension of the response examples
        :param method: string; 'knn', 'kde', 'kde_gen' or 'normal'. determines method used for estimating cross entropy
        :param hyperparams_dict: dict; dictionary of hyperparameters for density estimation. updated automatically
        :param name: string; name of the instance
        """
        super(DimReduce, self).__init__(name=name)

        if method not in ['knn', 'kde', 'kde_gen', 'normal']:
            raise NameError('Only one of knn, kde, kde_gen, and normal can be a valid estimation method')

        with self.name_scope:
            self.layers = layers
            self.d, self.m, self.p = d, m, p
            self.method = method  # determines which loss function is used
            self.seed = seed

            self.resnet = resnet
            if self.resnet:
                residual_init = tf.random.normal((self.m, self.d),
                                                 mean=0, stddev=.1, seed=self.seed, dtype=tf.float32)

                self.residual_transform = tf.Variable(residual_init, dtype=tf.float32, trainable=True,
                                                      name='residual_trans')

            if hyperparams_dict is not None:
                for param in hyperparams_dict:
                    if self.method == 'knn':
                        if param == 'k_z':  # number of nearest neighbors of estimation of density of dim reduced preds
                            self.k_z = hyperparams_dict[param]
                        if param == 'k_yz':  # number of nearest neighbors, est of joint density of dim reduced preds, resp
                            self.k_yz = hyperparams_dict[param]

                    if self.method == 'kde':
                        if param == 'h':  # kde smoothing parameter; don't worry about generalized kde for now
                            self.lambda_ = hyperparams_dict[param]

                    if self.method == 'kde_gen':
                        if param == 'A_inv':
                            self.A_inv = hyperparams_dict[param]  # smoothing matrix in generalized kde

    # @tf.Module.with_name_scope
    def to_lower_dims(self, x):
        """
        transform the data to the lower dimensional representation given current network params and architecture.
        essentially, the forwards pass up to the specifics of the loss function given by the 'method' parameter.

        :param x: tf.Variable; input matrix to be transformed
        :return: tf.Variable; matrix of low dimensional representations
        """

        if self.resnet:
            residual = tf.matmul(self.residual_transform, x)

        for layer in self.layers:
            x = layer.forward(x)

        x = x + residual if self.resnet else x
        return x

    @tf.Module.with_name_scope
    def update_hyperparams(self, hyperparams_dict):
        """
        update the density estimation hyperparameters of the instance. used in the setup of a new training phase.

        :param hyperparams_dict: dict; see exact contents in dim_reduce.py
        """

        self.batch_size = hyperparams_dict['batch_size']
        for param in hyperparams_dict:
            if self.method == 'knn':
                if param == 'k_z':  # number of nearest neighbors of estimation of density of dim reduced preds
                    self.k_z = hyperparams_dict[param]
                if param == 'k_yz':  # number of nearest neighbors, est of joint density of dim reduced preds, resp
                    self.k_yz = hyperparams_dict[param]

            if self.method == 'kde':
                if param == 'h':  # kde smoothing parameter; don't worry about generalized kde for now
                    self.lambda_ = hyperparams_dict[param]

            if self.method == 'kde_gen':
                if param == 'A_inv':
                    self.A_inv = hyperparams_dict[param]  # smoothing matrix in generalized kde

    @tf.Module.with_name_scope
    def loss_knn(self, x, y):
        """
        computes the loss when using k-nearest neighbors to estimate densities.

        :param x: tf.Variable; matrix containing higher dimensional predictor representations
        :param y: tf.Variable; raw responses
        :return: tf.Variable; loss estimate
        """

        batch_size = x.shape[1]

        Z = self.to_lower_dims(x)
        yZ = tf.transpose(tf.concat((y, Z), axis=0))

        V = tf.reshape(yZ, (batch_size, 1, self.p + self.m)) - tf.reshape(yZ, (1, batch_size, self.p + self.m))
        Vz = tf.reduce_prod(V[:, :, self.p:], axis=2)
        Vy = tf.reduce_prod(V[:, :, :self.p], axis=2)
        Vyz = Vz * Vy

        vol_z = -tf.math.top_k(-tf.abs(Vz), self.k_z + 1)[0][:, self.k_z]  # flexible to choice of k nearest neighbors
        vol_yz = -tf.math.top_k(-tf.abs(Vyz), self.k_yz + 1)[0][:, self.k_yz]
        est = (vol_z / vol_yz) * (self.k_yz / self.k_z)

        return tf.reduce_mean(-tf.math.log(est))

    @tf.Module.with_name_scope
    def loss_kde(self, x, y):

        """
        computes the loss when using kde to estimate densities.

        :param x: tf.Variable; matrix containing higher dimensional predictor representations
        :param y: tf.Variable; raw responses
        :return: tf.Variable; loss estimate
        """

        batch_size = x.shape[1]

        Zx = self.to_lower_dims(x)
        Zy = tf.transpose(tf.concat((y, Zx), axis=0))

        Z = tf.reshape(Zy, (batch_size, 1, self.m + self.p)) - \
            tf.reshape(Zy, (1, batch_size, self.m + self.p))
        Z2H = tf.math.square(Z) / tf.broadcast_to(tf.constant(self.lambda_, dtype='float32'),
                                                  [batch_size, batch_size, self.m + self.p])

        norms_z = tf.math.reduce_sum(Z2H[:, :, self.p:], axis=2)
        norms_y = tf.math.reduce_sum(Z2H[:, :, :self.p], axis=2)
        norms_yz = norms_z + norms_y  # marginally faster than 2 calls to tf.norm

        kde_z_ = tf.reduce_sum(tf.math.exp(-norms_z / 2), axis=0)
        kde_yz_ = tf.reduce_sum(tf.math.exp(-norms_yz / 2), axis=0)
        est = kde_yz_ * ((2 * np.pi) ** (-self.p / 2)) * (np.prod(self.lambda_[:self.p]) ** (-.5)) / kde_z_

        return tf.reduce_mean(-tf.math.log(est))

    @tf.Module.with_name_scope
    def loss_kde_gen(self, x, y):

        """
        computes the loss when using kde with a generalized covariance structure relating kernels to estimate densities.
        should not be used for standard kde (mixture of independent gaussians) as runs slower.

        :param x: tf.Variable; matrix containing higher dimensional predictor representations
        :param y: tf.Variable; raw responses
        :return: tf.Variable; loss estimate
        """

        batch_size = x.shape[1].value

        Zx = self.to_lower_dims(x)
        Zy = tf.transpose(tf.concat((y, Zx), axis=0))

        A_inv = tf.constant(self.A_inv)
        Z = tf.reshape(Zy, (batch_size, 1, self.m + self.p)) - tf.reshape(Zy, (1, batch_size, self.m + self.p))
        AZ = tf.einsum('lk...,ijk...->ijk...', A_inv, Z)

        ZAZ = Z * AZ

        norms_z = tf.math.reduce_sum(ZAZ[:, :, p:], axis=2)
        norms_y = tf.math.reduce_sum(ZAZ[:, :, :p], axis=2)
        norms_yz = tf.math.add(norms_z, norms_y)  # marginally faster than 2 calls to tf.norm

        kde_z_ = tf.reduce_sum(tf.exp(-norms_z / 2), axis=0)
        kde_yz_ = tf.reduce_sum(tf.exp(-norms_yz / 2), axis=0)
        est = kde_yz_ / kde_z_

        return tf.reduce_mean(-tf.math.log(est))

    @tf.Module.with_name_scope
    def loss_gauss(self, x, y):

        """
        computes the loss under the assumption that the data are joint gaussian

        :param x: tf.Variable; matrix containing higher dimensional predictor representations
        :param y: tf.Variable; raw responses
        :return: tf.Variable; loss estimate
        """

        batch_size = tf.constant(x.shape[1], dtype=tf.float32)

        Z = self.to_lower_dims(x)
        Z = tf.concat((y, Z), axis=0)

        mu = tf.reshape(tf.reduce_mean(Z, axis=1), [(self.m + self.p), 1])
        centered = tf.transpose(Z - mu)
        cov = tf.reduce_sum(tf.einsum('ij...,i...->ij...', centered, centered),
                            axis=0) / batch_size  # in lower dim space

        chol = tf.linalg.cholesky(cov[self.p:, self.p:])
        est = tf.linalg.det(cov[:self.p, :self.p] - tf.matmul(cov[:self.p, self.p:],
                                                      tf.linalg.cholesky_solve(chol, cov[self.p:, :self.p])))

        return est


    # def loss_variational(self, x, y):
    # # add in the loss used in dim_reduce_deep currently
    #
    #     # out0_1 = tf.matmul(trainable_params['T'], x1)
    #     # out0_2 = tf.matmul(trainable_params['T'], x2)
    #     #
    #     # yz1 = tf.concat([out0_1, y1], axis=0)
    #     # yz2 = tf.concat([out0_2, y1], axis=0)  # use ys from first batch, z from second
    #     # out1_1 = tf.nn.sigmoid(tf.matmul(trainable_params['W1'], yz1) + trainable_params['b1'])
    #     # out1_2 = tf.nn.sigmoid(tf.matmul(trainable_params['W1'], yz2) + trainable_params['b1'])
    #     # out2_1 = tf.nn.sigmoid(tf.matmul(trainable_params['W2'], out1_1) + trainable_params['b2'])
    #     # out2_2 = tf.nn.sigmoid(tf.matmul(trainable_params['W2'], out1_2) + trainable_params['b2'])
    #     # f_1 = tf.nn.sigmoid(tf.matmul(trainable_params['Wf'], out2_1))
    #     # f_2 = tf.nn.sigmoid(tf.matmul(trainable_params['Wf'], out2_2))
    #     #
    #     # f_1, f_2 = forward_pass_variational(trainable_params, X1, X2, Y1)
    #     #
    #     # loss = -tf.reduce_mean(f_1) + tf.log(tf.reduce_mean(tf.exp(f_2)))

    @tf.Module.with_name_scope
    def loss(self, x, y):
        """
        loss function conductor. handles which loss to use.
        :param x:
        :param y:
        :return:
        """

        if self.method == 'knn':
            loss = self.loss_knn(x, y)

        if self.method == 'kde':
            loss = self.loss_kde(x, y)

        if self.method == 'kde_gen':
            loss = self.loss_kde_gen(x, y)

        if self.method == 'normal':
            loss = self.loss_gauss(x, y)

        # if self.method == 'variational':
        #     loss = self.loss_variational(x, y)

        return loss


class LinearDimReduce(DimReduce):
    """
    class for dimension reduction with special methods for linear transformation learning. notice we don't check
    number of layers, as deep linear nets can still be useful.
    """

    def __init__(self, layers, d, m, p, method, hyperparams_dict=None, name="DimReduce"):
        """
        constructor

        :param layers: list of tf.Module subclassed layers (e.g. FullyConnected); specify the architecture
        :param d: int; original dimension of the predictor space
        :param m: int; dimension to which we're reducing predictors
        :param p: int; dimension of the response examples
        :param method: string; 'knn', 'kde', 'kde_gen' or 'normal'. determines method used for estimating cross entropy
        :param hyperparams_dict: dict; dictionary of hyperparameters for density estimation. updated automatically
        :param name: string; name of the instance
        """

        super(LinearDimReduce, self).__init__(layers, d, m, p, method, hyperparams_dict, name)

        invalid = 0
        for layer_num in range(len(self.layers)):

            # must be fully connected layer
            if not isinstance(self.layers[layer_num], FullyConnected):
                invalid = 1

            # some dimension checks
            if layer_num == 0:
                if self.layers[layer_num].input_dim != d:
                    invalid = 1

            if layer_num == (len(self.layers) - 1):
                if self.layers[layer_num].output_dim != m:
                    invalid = 1

            # check not affine
            print(self.layers[layer_num].b.trainable != 0)
            if (self.layers[layer_num].b.trainable != 0) or (np.any(self.layers[layer_num].b.numpy() != np.zeros(m))):
                invalid = 1

        if invalid:
            raise ValueError('\nYour proposed network architecture allows for the learning of a non-linear'
                             ' tranformation. \nThis class is specifically for learning of linear dimension reductions,'
                             ' so specify a linear '
                             'architecture and run again.\n')

    def current_transform(self):
        """
        returns the current linear transform specified by the network as a numpy array

        :return: np.ndarray; the transform
        """
        if len(self.layers) > 1:  # in the case of a deep linear network
            T = np.eye(self.d)
            for layer in self.layers:
                T = np.matmul(layer.T.numpy(), T)
        else:
            T = self.layers[0].T.numpy()

        return T


class FixedPartLinearDimReduce(DimReduce):
    """class for learning an m x d linear transformation with m x (d-m) trainable parameters; the leftmost m x m
     block of the transformation is the 'fixed_part'. only single layer learning is possible here. """

    def __init__(self, layers, d, m, p, method, fixed_part, hyperparams_dict=None, name="FixedDimReduce"):
        """
        contructor

        ::param layers: list of tf.Module subclassed layers (e.g. FullyConnected); must be a single linear layer
        :param d: int; original dimension of the predictor space
        :param m: int; dimension to which we're reducing predictors
        :param p: int; dimension of the response examples
        :param method: string; 'knn', 'kde', 'kde_gen' or 'normal'. determines method used for estimating cross entropy
        :param hyperparams_dict: dict; dictionary of hyperparameters for density estimation. updated automatically
        :param name: string; name of the instance
        """

        super(FixedPartLinearDimReduce, self).__init__(layers, d, m, p, method, hyperparams_dict, name)

        layer = self.layers[0]  # trainable part of the transformation

        invalid = False

        # check dimensions are correct
        if (layer.input_dim, layer.output_dim) != (d - m, m):
            invalid = True

        # check not affine
        if (layer.b.trainable != 0) or (np.any(layer.b.numpy() != np.zeros(m))):
            invalid = True

        # we don't allow for deep linear networks here
        if len(self.layers) != 1:
            invalid = True

        if invalid:
            raise ValueError('\nYour proposed network architecture does not force a single layer, fixed part learned '
                             'transformation. \nSpecify a valid architecture and run again.\n')

        self.fixed_part = tf.constant(fixed_part, dtype=tf.float32)
        self.T = 0


    # @tf.Module.with_name_scope
    def to_lower_dims(self, x):
        """
        override of DimReduce method for transforming to lower dimensions given the current network parameters
        :param x: tf.Variable; vector or more commonly, matrix whose columns are vector at which to eval PreLU
        :return: tf.Variable;
        """
        self.T = tf.concat((self.fixed_part, self.layers[0].T), axis=1)
        return tf.matmul(self.T, x)

    #@tf.Module.with_name_scope
    def current_transform(self):
        """
        returns the current learned transformation as an numpy array

        :return: np.ndarray; the current transformation
        """
        self.T = tf.concatenate((self.fixed_part, self.layers[0]), dtype=tf.float32)
        if save:
            np.save(path + '/T' + str(step) + '.npy', self.T.numpy())

        return self.T


##################################### UTILITIES FOR DATA READING/GENERATION  ###################################


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


def staircase_plot_simple(inverse_data, figname, ndim, scales, bin_num, response_names=None, pred_names=None):
    fig = plt.figure()
    fig.set_figwidth(10.0)
    fig.set_figheight(10.0)

    if response_names is not None and pred_names is not None:
        print(len(response_names))
        print(len(pred_names))
        print(ndim)
        assert (len(response_names) + len(pred_names) == ndim)

    for k in range(ndim):
        #print(k)
        ax = fig.add_subplot(ndim, ndim, 1 + k + k * ndim)
        ax.hist(inverse_data[:, k], density="True", facecolor='green', alpha=0.75, bins=bin_num)
        ax.yaxis.tick_right()
        if scales is not None:
            ax.set_xlim(scales[k])
        if k < ndim - 1:
            ax.tick_params(bottom=False, labelbottom=False)

        for kk in range(k):
            #print(kk)
            ax = fig.add_subplot(ndim, ndim, k * ndim + kk + 1)
            KDE = gaussian_kde(inverse_data[:, (kk, k)].T)
            z = KDE(inverse_data[:, (kk, k)].T)
            ax.scatter(inverse_data[:, kk], inverse_data[:, k], c=z, s=10, edgecolor='')
            if scales is not None:
                ax.set_xlim(scales[kk])
                ax.set_ylim(scales[k])
            if kk == 0:
                if response_names is not None and pred_names is not None:
                    if k < len(response_names):
                        ax.set_ylabel(response_names[k], fontsize=18)
                    else:
                        ax.set_ylabel(pred_names[k - len(response_names)], fontsize=18)
                else:
                    ax.set_ylabel("$x_{%d}$" % k, fontsize=18)
            else:
                ax.tick_params(left=False, labelleft=False)

            if k == ndim - 1:
                if response_names is not None and pred_names is not None:
                    if kk < len(response_names):
                        ax.set_xlabel(response_names[kk], fontsize=18)
                    else:
                        ax.set_xlabel(pred_names[kk - len(response_names)], fontsize=18)
                else:
                    ax.set_xlabel("$x_{%d}$" % kk, fontsize=18)
            else:
                ax.tick_params(bottom=False, labelbottom=False)

    fig.savefig(figname, format="png")
    fig.clf()
    plt.close()


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


################################### UTILITIES FOR TRAINING SETUP ######################################


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


##################################### UTILITIES FOR TRAINING EVALUATION #######################################

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

