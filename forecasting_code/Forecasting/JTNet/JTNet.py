# Support for Jacobian Tracker neural networks -- TensorFlow 2.0 version.
#
# Carlo Graziani, ANL

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
from scipy.special import gammaincinv
from scipy.stats import multivariate_normal as mvn

size = None


def setsize(sz):
    "Set input size"
    global size, tfsize
    size = sz
    tfsize = tf.constant(size)


def getsize():
    "Get input size"
    if not size:
        raise RuntimeError("Data vector size must be initialized first.")
    return size


########################################################################
########################################################################
########################################################################
class Linear_JT(tf.Module):
    """
    Dense linear layer, adapted for Jacobian Tracking.

    This is essentially a dense linear layer, with the following additional
    properties:

    (a) The weight parameters are stored in a square matrix, so the output
        has the same dimension as the input. This dimension is held in the
        JTNet module attribute 'size';

    (b)  The weights are not the weight parameters directly: rather,
        the matrix of parameters encodes the LU decomposition of the
        weight matrix, with the upper-triangular matrix U represented by
        the upper triangle of the parameter matrix (inclusive of the
        diagonal), and the lower unit triangular matrix L represented by
        the lower triangle of the parameter matrix (exclusive of the
        diagonal). The weight matrix is obtainable by "LU recomposition"
        of the factors (although in practice the computation performed
        is in fact L(Ux) -- which costs O(N^2) -- rather than (LU)x --
        which costs O(N^3));

    (c) The Log Jacobian of the transformation is stored as the tf.Variable LogJac.

    Constructor Args:

      name (string): Name of current layer, used to establish the name scope

      kernel_initializer (None or tf.initializer): Initializer method for the "weight"
        matrix.

      initial_bias (None or tf.constant): Array to initialize the biases

      bw (None or int): Bandwidth of the desired weight matrix. By default the
        full matrix is used, but if bw<size-1 then a banded diagonal matrix with
        this bandwidth is used. bw=0 corresponds to a diagonal matrix.

    """

    ########################################################################
    def __init__(self, name, kernel_initializer=None, initial_bias=None, bw=None, seed=None):
        """
        Constructor
        """
        super(Linear_JT, self).__init__(
            name=name)  # https://www.tensorflow.org/api_docs/python/tf/Module#with_name_scope
        size = getsize()

        if not bw:
            self.bw_ = size - 1
        elif bw < 0 or bw >= size:
            raise ValueError("Must set bw such that 0 <= bw < %d" % size)

        with self.name_scope:
            if kernel_initializer is None:
                #stdev = np.sqrt(1. / size)
                #stdev = .1
                stdev = np.sqrt(1. / size)
                kernel_initializer_weight = tf.random.normal((size, size), stddev=stdev, seed=seed, dtype=tf.float64)
                # t_dist = tfp.distributions.StudentT(2, loc=0, scale=.5)
                # kernel_initializer_weight = t_dist.sample((size, size), seed=seed)
                kernel_initializer, _ = tf.linalg.lu(kernel_initializer_weight)

            if initial_bias is None:
                initial_bias = tf.zeros([size, 1], dtype=tf.float64)

            self.b = tf.Variable(initial_bias, name="bias")
            self.LU = tf.Variable(kernel_initializer, name="LU")

            self.upper_ones_ = tf.constant(np.triu(np.ones_like(self.LU.numpy()), 0), dtype=tf.float64)  # 'private' attribute
            self.lower_ones_ = tf.constant(np.tril(np.ones_like(self.LU.numpy()), -1), dtype=tf.float64)
            self.size = (size, size)

            self.L, self.U, self.Weights, self.LogJac_0 = None, None, None, None  # initialize other useful attributes


    ########################################################################

    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Applies the decomposed linear layer, and tracks the Jacobian.

        Args:
            y (tensor): Batch of output vectors.

        Returns:
            x (tensor) Input vectors
        """

        # Note, due to right-multiplication, the weights matrix must be transposed
        # in order to maintain the consistent interpretation of "LU"
        with tf.name_scope("inverse"):

            self.U = self.upper_ones_ * self.LU
            self.L = self.lower_ones_ * self.LU + tf.eye(self.LU.numpy().shape[0], dtype=tf.float64)

            xx = tf.transpose(y - tf.transpose(self.b))
            yy = tf.linalg.triangular_solve(self.L, xx, lower=True, name='lower')
            x = tf.linalg.triangular_solve(self.U, yy, lower=False, name='upper')

            self.LogJac_0 = tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(self.U))))  #J(phi)
            self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(y)[0:1], name="LogJac")

        return tf.transpose(x)

    ########################################################################

    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Forward transformation.

        Args:
            x (tensor): Batch of "input" vectors.

        Returns:
            y (tensor): Batch of "output" vectors.

        """
        with tf.name_scope("forward"):

            self.U = self.upper_ones_ * self.LU
            self.L = self.lower_ones_ * self.LU + tf.eye(self.LU.numpy().shape[0], dtype=tf.float64)

            #self.Weights = tf.matmul(self.L, self.U)

            out = tf.linalg.matmul(tf.linalg.matmul(x, self.U, transpose_b=True), self.L, transpose_b=True) + self.b

            self.LogJac_0 = tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(self.U))))
            self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(x)[0:1], name="LogJac")

        return out


    @tf.Module.with_name_scope
    def get_weights(self):
        """
        access the weights of the layer. note that weight matrix is not explicitly computed unless this method is
        called

        :return:
        """

        return tf.matmul(self.L, self.U)


########################################################################
########################################################################
########################################################################

# class Radial_JT(tf.Module):
#     """
#     Radial Transform -
#
#     We wish to transform such that the radial (r=sqrt(sum(x'^2))) pdf transforms from
#
#     f_r' = r'^(n/2-1)e^(-r'^2)
#
#     to (given scaling constant s)
#
#           ---
#     f_r = |s*(r/a)^(n-1),                    0<r<b
#           |s*a^(-(n-1))*b^(c+1)r^(-c),         r>b
#           ---
#
#     To do so, we first transform from normal to uniform via the cdf of the variable
#
#     u = igamma(n/2,r')
#
#     then using the inverse cdf of the desired transformed cdf
#
#     F_r =
#
#     The random variable is then transformed from input random variable x = (r/r')x'
#
#     Constructor Args:
#
#       name (string): Name of current layer, used to establish the name scope
#
#       parameter_initializer (None or tf.initializer): (3 parameters)
#     """
#
#     ########################################################################
#     def __init__(self, name, b_initializer=None, c_initializer=None):
#         """
#         Constructor
#         """
#         super(Radial_JT, self).__init__(name=name)
#         size = getsize()
#         with self.name_scope:
#             if b_initializer is None:
#                 b_initializer = tf.constant(0.)
#             if c_initializer is None:
#                 c_initializer = tf.constant(0.)
#             self.b = tf.math.exp(tf.Variable(b_initializer, name="b"))
#             self.c = (1 + tf.math.exp(tf.Variable(c_initializer, name="c")))
#
#             self.scaling = (self.b) ** (size) * (1. / size + 1 / (self.c - 1.))
#             self.fitfac = self.b ** (size - 1 + self.c)
#             self.pwint = ((self.b ** size / self.scaling) / size)
#
#             # Create grid for inverse incomplete gamma function
#             u = np.linspace(0, .99999, 10000)
#             self.gammaincinv = gammaincinv(size / 2, u)
#
#     ########################################################################
#     @tf.Module.with_name_scope
#     def forward(self, x):
#         """
#         Applies the nonlinear layer, and tracks the Jacobian.
#
#         Args:
#             x (tensor): Batch of input vectors.
#
#         Returns:
#             y (tensor) Output of the linear layer.
#         """
#
#         with tf.name_scope("forward"):
#             r = tf.math.sqrt(tf.math.reduce_sum(x ** 2, axis=1))
#
#             growpart = (1 / self.scaling) * (r ** size / size)
#             decaypart = self.pwint \
#                         + (self.fitfac / self.scaling) * (self.b ** (1 - self.c) / (self.c - 1) \
#                                                           - r ** (1 - self.c) / (self.c - 1))
#
#             u = tf.where(r < self.b, growpart, decaypart)
#             rp = tfp.math.interp_regular_1d_grid(u, 0, 1, self.gammaincinv)
#
#             xp = x * tf.math.expand_dims(rp / r, axis=1)
#
#             growpdf = (1 / self.scaling) * r ** (size - 1)
#             decaypdf = (self.fitfac / self.scaling) * r ** (-self.c)
#
#             logdudr = tf.math.log(tf.where(r < self.b, growpdf,
#                                            decaypdf))  # was a negative sign in front removed as im not sure why its there
#             logdudrp = (size / 2 - 1) * tf.math.log(rp) - rp - tf.math.lgamma(size / 2)
#             self.LogJac = (size - 1) * (tf.math.log(rp / r)) + logdudr - logdudrp
#
#         return xp
#
#     ########################################################################
#     @tf.Module.with_name_scope
#     def inverse(self, xp):
#         """
#         Inverse transformation.
#
#         Args:
#
#             y (tensor): Batch of "output" vectors.
#
#         Returns:
#             x (tensor): Batch of "input" vectors.
#
#         """
#
#         with tf.name_scope("inverse"):
#             rp = tf.math.sqrt(tf.math.reduce_sum(xp ** 2, axis=1))
#             u = tf.math.igamma(size / 2, rp)
#
#             uabove = tf.math.maximum(u, self.pwint)  # prevents problems with tf.where on NaN
#
#             growpart = (self.scaling * u * size) ** (1. / size)
#             decaypart = self.b / ((1 - uabove) * (self.c + size - 1) / size) ** (1 / (self.c - 1))
#             r = tf.where(u < self.pwint, growpart, decaypart)
#             x = xp * tf.expand_dims(r / rp, axis=1)
#
#             growpdf = (1 / self.scaling) * r ** (size - 1)
#             decaypdf = (self.fitfac / self.scaling) * r ** (-self.c)
#
#             logdudr = tf.math.log(tf.where(r < self.b, growpdf, decaypdf))
#             logdudrp = (size / 2 - 1) * tf.math.log(rp) - rp - tf.math.lgamma(size / 2)
#             self.LogJac = (size - 1) * (tf.math.log(rp / r)) + logdudr - logdudrp
#
#         return x


########################################################################
########################################################################
########################################################################

# class NonLinear_Squared_JT(tf.Module):
#
#
#     def __init__(self, name, kernel_initializer=None, seed=None):
#         """
#         constructor
#         :param name:
#         :param kernel_initializer:
#         :param seed:
#         """
#
#         super(NonLinear_Squared_JT, self).__init__(name=name)
#         size = getsize()
#
#         with self.name_scope:
#             if kernel_initializer is None:

########################################################################
########################################################################
########################################################################

class Nonlinear_Squared_Coupling(tf.Module):
    """
    A class for implementing a coupling flow in the style of https://arxiv.org/pdf/1901.10548.pdf
    (using the nonlinear-squared coupling function).
    """

    ########################################################################
    def __init__(self, name, theta_depth, resnet, use_subresiduals, subresidual_skip_num,
                                                      train_subresiduals, trans_comps=None, seed=None):
        """
        Constructor
        :param name: string; name of the flow
        :param trans_comps: list of ints; coordinates of the components to be transformed nonlinearly by the flow (i.e.
        the coordinates of 'x_A' as the are called by https://arxiv.org/abs/1908.09257, sec 3.4.1.
        :param seed: int; seed
        """
        super(Nonlinear_Squared_Coupling, self).__init__(name=name)
        size = getsize()  # dimension of the input/output

        with self.name_scope:

            self.seed = 1 if seed is None else seed

            # attributes relating to which variables get transformed
            self.trans_comps = list(range(int(np.floor(size/2)))) if trans_comps is None else trans_comps
            self.perm = tf.constant(self.trans_comps + [i for i in range(size) if i not in self.trans_comps], dtype=tf.int32)
            self.inverse_perm = tf.constant(list(np.argsort(np.array(self.perm))), dtype=tf.int32)  # inv takes us back to order of comps before perm

            self.size_trans = len(self.trans_comps)
            self.size_id = size - self.size_trans
            self.NUM_PSEUDOPARAMS = 5   # this is a constant of this particular coupling

            # the conditioner network
            self._theta = _Theta(name='Theta', num_layers=theta_depth, input_dim=self.size_id, output_dim=self.NUM_PSEUDOPARAMS,
                                 size_id=self.size_id, size_trans=self.size_trans, resnet=resnet,
                                 use_subresiduals=use_subresiduals, subresidual_skip_num=subresidual_skip_num,
                                 train_subresiduals=train_subresiduals, seed=self.seed)

            # attributes relating to pseudoparams
            self.ALPHA = .95  # a tuning param
            self.const = tf.constant((8 * np.sqrt(3) * self.ALPHA / 9), dtype=tf.float64)  # a useful constant
            self.a, self.b, self.c, self.d, self.g = 0, 1, 0, 1, 0

            # attributes relating to jacobian tracking
            self.jacobian_diagonal = tf.ones(size, dtype=tf.float64)
            self.LogJac_0, self.LogJac = 0, 0

    ########################################################################

    @tf.Module.with_name_scope
    def _update_pseudo_params(self, x_b):
        """
        Map the components that are subjected to identity transformation to pseudoparameter space.

        :param x_b: N x self.size_id tf.Variable; the components to mapped
        :return: N x 5 x self.size_trans tf.Variable; 5-tuple of pseduoparams for each component that's transformed for
        each example in the batch.
        """

        theta_forwards = self._theta.forward(x_b)

        self.a, log_b, c_prime, log_d, self.g = theta_forwards[:, 0, :], theta_forwards[:, 1, :], \
                                                theta_forwards[:, 2, :], theta_forwards[:, 3, :], theta_forwards[:, 4,
                                                                                                  :]
        self.b, self.d = tf.math.exp(log_b), tf.math.exp(log_d)
        self.c = self.const * (self.b / self.d) * tf.math.tanh(c_prime)

        return None


    ########################################################################

    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Forward pass of the coupling flow.

        :param y: N x (p+m) tf.Variable; 'input' of a coupling flow
        :return: N x (p+m) tf.Variable; 'output' of a coupling flow
        """

        x = tf.gather(x, indices=self.perm, axis=1)  # permute the columns to use existing architecture for trans
        x_a, x_b = x[:, :self.size_trans], x[:, self.size_trans:]  # partition of the input variables

        # give N x 5 x self.size_trans tensor with 5 pseudoparams for each sample and each x_A com
        self._update_pseudo_params(x_b)

        y_a = self.b * x_a + self.a + self.c / (1 + (self.d * x_a + self.g)**2)

        # compute partials derivatives on diagonal of Jacobian J ; these specify the determinant of J
        jac_diag = self.b - 2 * self.c * self.d * (self.d * x_a + self.g) / ((1 + (self.d * x_a + self.g)**2)**2)
        self.jacobian_diagonal = tf.concat((jac_diag, tf.ones((x.shape[0], self.size_id), dtype=tf.float64)), axis=1)

        y = tf.concat((y_a, x_b), axis=1)
        y = tf.gather(y, self.inverse_perm, axis=1)  # shift the components back to their original order

        self.LogJac_0 = tf.reduce_sum(tf.math.log(self.jacobian_diagonal), axis=1)  # addition commutes
        self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(x)[0:1], name="LogJac")

        return y


    ###########################################################

    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Inversion of the coupling flow. This requires polynomial root finding via Cardano's formula. See Supplementary
        Materials 'C' of https://arxiv.org/pdf/1901.10548.pdf for details. bottom of these can get nans
        :param y: N x (p+m) tf.Variable; 'output' of a coupling flow
        :return: N x (p+m) tf.Variable; 'input' of a coupling flow
        """

        y = tf.gather(y, indices=self.perm, axis=1)  # put comps into the order in which they were fed to the trans
        y_a, y_b = y[:, :self.size_trans], y[:, self.size_trans:]  # partition; y_b = x_b
        self._update_pseudo_params(y_b)  # y_b = x_b -> we can get pseudo params for the output y

        # some constants defining the relevant cubic equation that must be solved to invert
        a_tilde = -self.b * self.d ** 2
        b_tilde = (y_a - self.a) * self.d ** 2 - 2 * self.d * self.g * self.b
        c_tilde = 2 * self.d * self.g * (y_a - self.a) - self.b * (self.g ** 2 + 1)
        d_tilde = (y_a - self.a) * (self.g ** 2 + 1) - self.c

        # use Cardano's formula to solve the relevant equation
        q = (3 * a_tilde * c_tilde - b_tilde ** 2) / (9 * a_tilde ** 2)
        r = (9 * a_tilde * b_tilde * c_tilde - 27 * a_tilde ** 2 * d_tilde - 2 * b_tilde ** 3) / (54 * a_tilde ** 3)

        # apparently tf can't handle cube roots of negative numbers, so do some hacking
        discriminant = q ** 3 + r ** 2  # should be positive, guaranteeing exactly 1 real root, and thus invertibility

        s_cubed = r + discriminant ** (1 / 2)
        s_sign = tf.sign(s_cubed)  # signs of the s's
        s_abs = (tf.abs(s_cubed)) ** (1 / 3)
        s = s_sign * s_abs

        t_cubed = r - discriminant ** (1 / 2)
        t_sign = tf.sign(t_cubed)  # signs of the t's
        t_abs = (tf.abs(t_cubed)) ** (1 / 3)
        t = t_sign * t_abs

        # compute the unique real root (inverse of y_a)
        x_a = s + t - b_tilde / (3 * a_tilde)  # the inverse of the transformed part

        x = tf.concat((x_a, y_b), axis=1)
        x = tf.gather(x, self.inverse_perm, axis=1)

        # compute partials derivatives on diagonal of Jacobian J ; these specify the determinant of J
        jac_diag = self.b - 2 * self.c * self.d * (self.d * x_a + self.g) / ((1 + (self.d * x_a + self.g) ** 2) ** 2)
        self.jacobian_diagonal = tf.concat((jac_diag, tf.ones((x.shape[0], self.size_id), dtype=tf.float64)), axis=1)

        self.LogJac_0 = tf.reduce_sum(tf.math.log(self.jacobian_diagonal), axis=1)
        self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(y)[0:1], name="LogJac")

        return x


########################################################################
########################################################################
########################################################################

class _Theta(tf.Module):
        """
        Class for transforming inputs into pseduo-parameters for a coupling flow, as in section 3.4.1 of
         https://arxiv.org/abs/1908.09257. There is no need for Jacobian tracking here, as the the determinant
         of the Jacobian depends only on Theta through it's output.
        """

        def __init__(self, name, num_layers, input_dim, output_dim, size_trans, size_id, resnet=True,
                     use_subresiduals=False, subresidual_skip_num=4, train_subresiduals=False, seed=None):
            """
            Constructor

            :param name: string; name of the instance
            :param num_layers: int; number of layers in total in Theta (a layer here is a linear, bias, activation
            triple)
            :param input_dim: int; dimension of the components that map to pseduoparam space in the flow
            :param output_dim: int; number of pseudo params for the coupling bijection h. in our code for now, h is
            fixed and has 5 psuedoparams
            :param size_trans: int; number of components being transformed by the coupling flow
            :param size_id: int; number of components pushed through the flow via identity transform
            :param seed: int; seed
            """

            super(_Theta, self).__init__(name=name)

            with self.name_scope:

                self.seed = seed if seed is not None else 1
                self.name = name
                self.num_layers = num_layers
                self.input_dim = input_dim
                self.output_dim = output_dim  # this is self.NUM_PSEUDOPARAMS = 5

                self.size_trans = size_trans
                self.size_id = size_id

                self.resnet = resnet
                self.use_subresiduals = use_subresiduals  # subresiduals make stability problems in some networks
                self.subresidual_skip_num = subresidual_skip_num  # number of 3-tuple layers the subresiduals skip
                self.train_subresiduals = train_subresiduals

            if self.resnet:
                residual_init = tf.random.normal((self.output_dim, self.input_dim, self.size_trans),
                                                   mean=0, stddev=.1, seed=self.seed, dtype=tf.float64)

                self.residual_transform = tf.Variable(residual_init, dtype=tf.float64, trainable=True,
                                                      name='residual_trans')
                if self.use_subresiduals:
                    self.subresiduals = []
                    num_subresiduals = int(np.floor(num_layers/self.subresidual_skip_num))

                    for sub in range(num_subresiduals):
                        if sub == 0:  # first one has to map to a different dimensional space in general
                            subresidual_init = tf.random.normal((self.output_dim, self.input_dim, self.size_trans),
                                                   mean=0, stddev=.05, seed=self.seed+1, dtype=tf.float64)
                            subresidual_transform = tf.Variable(subresidual_init, dtype=tf.float64, trainable=True,
                                                              name='subresidual{}'.format(sub))
                        else:
                            subresidual_init = tf.reshape(tf.repeat(tf.eye(self.output_dim, dtype=tf.float64), self.size_trans),
                                                    (self.output_dim, self.output_dim, self.size_trans))
                            subresidual_transform = tf.Variable(subresidual_init, dtype=tf.float64,
                                                                trainable=self.train_subresiduals,
                                                              name='subresidual{}'.format(sub))
                        self.subresiduals += [subresidual_transform]

            self.layers = []
            for i in range(num_layers):

                if i == 0:
                    linear_init = tf.random.normal((self.output_dim, self.input_dim, self.size_trans),
                                                   mean=0, stddev=.05, seed=self.seed, dtype=tf.float64)
                else:
                    linear_init = tf.random.normal((self.output_dim, self.output_dim, self.size_trans),
                                                   mean=0, stddev=.05, seed=self.seed, dtype=tf.float64)

                linear = tf.Variable(linear_init, dtype=tf.float64, trainable=True, name='linear{}'.format(i))
                bias = tf.Variable(tf.zeros((self.output_dim, self.size_trans), dtype=tf.float64),  trainable=True,
                                   name='bias{}'.format(i))

                self.layers += [[linear, bias, tf.nn.relu]]

        def forward(self, x):
            """
            Forward pass for the theta network. Much structure is hard coded in here.

            :param x: N x (m+p) tf.Variable; batch of examples. This should be x_B from the paper cited above.
            :return: N x 5 x size_trans tf Variable; returns the pseudo param 5-tuple for each component to be
            transformed for each example in the batch, up to some final transformations.
            """

            N = x.shape[0]

            if self.resnet:
                residual = tf.einsum('ijk, lj -> lik', self.residual_transform, x)
                subresidual = None

            for layer_num in range(0, self.num_layers, 1):  # each 'layer' is a (linear, bias, relu) triple

                if self.use_subresiduals and layer_num % self.subresidual_skip_num == 0:
                    x = x + subresidual if subresidual is not None else x
                    if layer_num != (self.num_layers - self.num_layers % self.subresidual_skip_num):  # don't transform the last time
                        if layer_num == 0:
                            subresidual = tf.einsum('ijk, lj -> lik',
                                                    self.subresiduals[int(layer_num/self.subresidual_skip_num)], x)
                        else:
                            subresidual = tf.einsum('ijk, ljk -> lik',
                                                    self.subresiduals[int(layer_num/self.subresidual_skip_num)], x)

                if layer_num == 0:
                    x = tf.einsum('ijk, lj -> lik', self.layers[layer_num][0], x)
                else:  # after the first transformation, the data are 3d
                    x = tf.einsum('ijk, ljk -> lik', self.layers[layer_num][0], x)

                x = x + tf.broadcast_to(self.layers[layer_num][1], (N, self.output_dim, self.size_trans))

                x = self.layers[layer_num][2](x)  # every 3rd 'layer' is an activation function

            x = residual + x if self.resnet else x  # add in skip connection

            return x


########################################################################
########################################################################
########################################################################

class Triang_NonLinear_JT(tf.Module):
    """
    Dense Nonlinear Triangular layer, adapted for Jacobian Tracking.


    Constructor Args:

      name (string): Name of current layer, used to establish the name scope

      kernel_initializer (None or tf.initializer): Initializer method for the "weight"
        matrix.

    """

    ########################################################################
    def __init__(self, name, kernel_initializer=None, seed=None):
        """
        Constructor
        """
        super(Triang_NonLinear_JT, self).__init__(name=name)
        size = getsize()
        with self.name_scope:
            if kernel_initializer is None:
                #stdev = np.sqrt(1. / size)
                stdev = .1
                kernel_initializer = tf.random.normal((size, size), stddev=stdev, seed=seed)
            self.A = tf.math.exp(tf.Variable(kernel_initializer, name="A"))

            # We will have to broadcast this when we know the number of samples
            self.LogJac_0 = tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(self.A))))

    ########################################################################
    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Applies the nonlinear layer, and tracks the Jacobian.

        Args:
            x (tensor): Batch of input vectors.

        Returns:
            y (tensor) Output of the linear layer.
        """

        # Has to treat the extreme behavior via an asymptotic limit

        with tf.name_scope("forward"):
            y = tf.expand_dims(self.A[0, 0] ** (-1) * x[:, 0], 1)
            for i in range(1, size):
                first = x[:, i]
                a = self.A[i, 0:i]
                b = self.A[0:i, i]
                temp_y = tf.where(tf.math.abs(y[:, 0:i]) > 40, 0.0 * y[:, 0:i], y[:, 0:i])
                temp_y2 = tf.math.asinh(tf.math.sinh(temp_y) * a + b)
                approx = y[:, 0:i] + tf.math.log(a)
                fixed = tf.where(tf.math.abs(y[:, 0:i]) > 40.0, approx, temp_y2)
                su = (first - tf.math.reduce_sum(fixed, axis=1)) * self.A[i, i] ** (-1)
                su = tf.expand_dims(su, 1)
                y = tf.concat([y, su], axis=1)

            self.LogJac_0 = tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(self.A))))
            self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(x)[0:1], name="LogJac")

        return y

    ########################################################################
    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Inverse transformation.

        Args:

            y (tensor): Batch of "output" vectors.

        Returns:
            x (tensor): Batch of "input" vectors.

        """
        with tf.name_scope("inverse"):
            x = tf.expand_dims(self.A[0, 0] * y[:, 0], 1)
            for i in range(1, size):
                a = self.A[i, 0:i]
                b = self.A[0:i, i]
                temp_y = tf.where(tf.math.abs(y[:, 0:i]) > 40, 0.0 * y[:, 0:i], y[:, 0:i])
                temp_x = tf.asinh(a * tf.math.sinh(temp_y) + b)
                approx = y[:, 0:i] + tf.math.log(a)
                fixed = tf.where(tf.math.abs(y[:, 0:i]) > 40.0, approx, temp_x)
                first = self.A[i, i] * y[:, i]
                # fixed = (a*tf.asinh(y[:,0:i]/a)+b*tf.tanh(y[:,0:i]/b))/2
                su = first + tf.math.reduce_sum(fixed, axis=1)
                su = tf.expand_dims(su, 1)
                x = tf.concat([x, su], axis=1)

            self.LogJac_0 = tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(self.A))))
            self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(y)[0:1], name="LogJac")

        return x


########################################################################
########################################################################
########################################################################

class PReLU_JT(tf.Module):
    """
    PReLU Activation, adapted to Jacobian Tracking

    This is the bog-standard PReLU activation, with the additional feature
    that the Jacobian of the activation is stored in self.LogJac. The function
    is, element-wise

           | exp(alpha)*x for x<0
    f(x) = |
           | x for x>0

    where alpha is a vector of dimension equal to the JTNet module attribute 'size'.

    Constructor Args:

      name (string): Name of current layer, used to establish the name scope

      alpha_initializer (None or tf.initializer): Initializer method for alpha array

    """

    ########################################################################
    def __init__(self, name, alpha_initializer=None):
        """
        Constructor
        """
        super(PReLU_JT, self).__init__(name=name)
        with self.name_scope:
            if alpha_initializer is None:
                #alpha_initializer = -1.0 * tf.ones(size)
                alpha_initializer = tf.zeros(size, dtype=tf.float64)

            self.alpha = tf.Variable(alpha_initializer, name="alpha")
            self.slopes = tf.math.exp(self.alpha, name="slopes")

    ########################################################################
    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Applies the PReLU activation and stores the Jacobian

        Args:
            x (tensor): Batch of input vectors

        Returns:
            y (tensor): Output of the PReLU operation

        """

        with tf.name_scope("forward"):
            self.slopes = tf.math.exp(self.alpha)
            y = tf.where(x > 0, x, x * self.slopes)

            zs = tf.zeros_like(x)
            os = tf.ones_like(x)
            lj = tf.where(x > 0, zs, self.alpha * os)
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return (y)

    ########################################################################
    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Inverse transformation
        Args:

            y (tensor): Batch of "output" vectors.

        Returns:
            x (tensor): Batch of "input" vectors.

        """

        with tf.name_scope("inverse"):
            self.slopes = tf.math.exp(self.alpha)  # update slopes attribute
            x = tf.where(y > 0, y, y / self.slopes)  # why the gradient seems to get through where here?

            zs = tf.zeros_like(x)
            os = tf.ones_like(x)
            lj = tf.where(x > 0, zs, self.alpha * os)
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return x


########################################################################

class DPReLU_JT(tf.Module):
    """
    DPReLU Activation, adapted to Jacobian Tracking

    This is the double PReLU activation, with the additional feature
    that the Jacobian of the activation is stored in self.LogJac. The function
    is, element-wise

           | exp(alpha)*x for x<0
    f(x) = |
           | exp(beta)x for x>0

    where alpha is a vector of dimension equal to the JTNet module attribute 'size'.

    Constructor Args:

      name (string): Name of current layer, used to establish the name scope

      alpha_initializer (None or tf.initializer): Initializer method for alpha array

    """

    ########################################################################
    def __init__(self, name, alpha_initializer=None, beta_initializer=None):
        """
        Constructor
        """

        super(DPReLU_JT, self).__init__(name=name)
        with self.name_scope:

            if alpha_initializer is None:
                alpha_initializer = -1.0 * tf.ones(size, dtype=tf.float64)

            if beta_initializer is None:
                beta_initializer = 0.0 * tf.ones(size, dtype=tf.float64)

            self.alpha = tf.Variable(alpha_initializer, name="alpha")
            self.slopes = tf.math.exp(self.alpha, name="slopes")

            self.beta = tf.Variable(beta_initializer, name="beta")
            self.slopesB = tf.math.exp(self.beta, name="slopesB")

    ########################################################################
    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Applies the DPReLU activation and stores the Jacobian

        Args:
            x (tensor): Batch of input vectors

        Returns:
            y (tensor): Output of the DPReLU operation

        """

        with tf.name_scope("forward"):
            self.slopes = tf.math.exp(self.alpha)
            self.slopesB = tf.math.exp(self.beta)

            y = tf.where(x > 0, x * self.slopesB, x * self.slopes)
            os = tf.ones_like(x)
            lj = tf.where(x > 0, self.beta * os, self.alpha * os)
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return y

    ########################################################################
    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Inverse transformation
        Args:

            y (tensor): Batch of "output" vectors.

        Returns:
            x (tensor): Batch of "input" vectors.

        """

        with tf.name_scope("inverse"):
            self.slopes = tf.math.exp(self.alpha)
            self.slopesB = tf.math.exp(self.beta)

            x = tf.where(y > 0, y / self.slopesB, y / self.slopes)

            os = tf.ones_like(y)
            lj = tf.where(y > 0, self.beta * os, self.alpha * os)
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return x


########################################################################
########################################################################
########################################################################

class DIPReLU_JT(tf.Module):
    """
    DIPReLU Activation, adapted to Jacobian Tracking.

    This is the double identity PReLU activation, with the additional feature
    that the Jacobian of the activation is stored in self.LogJac. The function
    is, element-wise

           | x+(exp(alpha_2)-1)*exp(beta) for   exp(beta)<x
    f(x) = | exp(alpha_2)x                for           0<x<exp(beta)
           | exp(alpha_1)x                for  -exp(beta)<x<0
           | x-(exp(alpha_1)-1)*exp(beta) for             x<-exp(beta)

    where alpha is a vector of dimension equal to the JTNet module attribute 'size'.

    Constructor Args:

      name (string): Name of current layer, used to establish the name scope

      alpha_initializer (None or tf.initializer): Initializer method for alpha array

    """

    ########################################################################
    def __init__(self, name, alpha_1_initializer=None, alpha_2_initializer=None,
                 beta_initializer=None):
        """
        Constructor
        """

        super(DIPReLU_JT, self).__init__(name=name)
        with self.name_scope:

            if alpha_1_initializer is None:
                alpha_1_initializer = 0.0 * tf.ones(size, dtype=tf.float64)

            if alpha_2_initializer is None:
                alpha_2_initializer = 0.0 * tf.ones(size, dtype=tf.float64)

            if beta_initializer is None:
                beta_initializer = 0.0 * tf.ones(size, dtype=tf.float64)

            self.alpha_1 = tf.Variable(alpha_1_initializer, name="alpha_1")
            self.slopes_1 = tf.math.exp(self.alpha_1, name="slopes_1")

            self.alpha_2 = tf.Variable(alpha_2_initializer, name="alpha_2")
            self.slopes_2 = tf.math.exp(self.alpha_2, name="slopes_2")

            self.beta = tf.Variable(beta_initializer, name="beta")
            self.hshift = tf.math.exp(self.beta, name="hshift")

            self.vshift_1 = (self.slopes_1 - 1.0) * self.hshift
            self.vshift_2 = (self.slopes_2 - 1.0) * self.hshift
            self.yhshift_1 = self.vshift_1 + self.hshift
            self.yhshift_2 = self.vshift_2 + self.hshift

            self.LogJac = 0

    ########################################################################
    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Applies the DIPReLU activation and stores the Jacobian

        Args:
            x (tensor): Batch of input vectors

        Returns:
            y (tensor): Output of the DIPReLU operation

        """

        with tf.name_scope("forward"):

            self.slopes_1 = tf.math.exp(self.alpha_1, name="slopes_1")
            self.slopes_2 = tf.math.exp(self.alpha_2, name="slopes_2")

            self.hshift = tf.math.exp(self.beta, name="hshift")
            self.vshift_1 = (self.slopes_1 - 1.0) * self.hshift
            self.vshift_2 = (self.slopes_2 - 1.0) * self.hshift

            yu = x + self.vshift_2
            ym_2 = self.slopes_2 * x
            ym_1 = self.slopes_1 * x
            yl = x - self.vshift_1

            y1 = tf.where(x < -self.hshift, yl, ym_1)
            y2 = tf.where(x > 0, ym_2, y1)
            y = tf.where(x > self.hshift, yu, y2)

            zs = tf.zeros_like(y)
            os = tf.ones_like(x)
            lj1 = tf.where(x < -self.hshift, zs, self.alpha_1 * os)
            lj2 = tf.where(x > 0, self.alpha_2 * os, lj1)
            lj = tf.where(x > self.hshift, zs, lj2)

            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return y

    ########################################################################
    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Inverse transformation
        Args:

            y (tensor): Batch of "output" vectors.

        Returns:
            x (tensor): Batch of "input" vectors.

        """

        with tf.name_scope("inverse"):
            self.slopes_1 = tf.math.exp(self.alpha_1, name="slopes_1")
            self.slopes_2 = tf.math.exp(self.alpha_2, name="slopes_2")

            self.hshift = tf.math.exp(self.beta, name="hshift")
            self.vshift_1 = (self.slopes_1 - 1.0) * self.hshift
            self.vshift_2 = (self.slopes_2 - 1.0) * self.hshift
            self.yhshift_1 = self.vshift_1 + self.hshift
            self.yhshift_2 = self.vshift_2 + self.hshift

            xu = y - self.vshift_2
            xm_2 = y / self.slopes_2
            xm_1 = y / self.slopes_1
            xl = y + self.vshift_1

            x1 = tf.where(y < -self.yhshift_1, xl, xm_1)
            x2 = tf.where(y > 0, xm_2, x1)
            x = tf.where(y > self.yhshift_2, xu, x2)

            zs = tf.zeros_like(y)
            os = tf.ones_like(x)
            lj1 = tf.where(x < -self.hshift, zs, self.alpha_1 * os)
            lj2 = tf.where(x > 0, self.alpha_2 * os, lj1)
            lj = tf.where(x > self.hshift, zs, lj2)
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return x


########################################################################
########################################################################
########################################################################

class Sinh_JT(tf.Module):
    """
    Hyperbolic Sine Activation, adapted to Jacobian Tracking.

    This is the sinh activation, with the additional feature
    that the Jacobian of the activation is stored in self.LogJac. The function
    is, element-wise

    f(x) = e^(alpha) * sinh( x * e^(-alpha) )


    where alpha is a vector of dimension equal to the JTNet module attribute 'size'.

    Constructor Args:

      name (string): Name of current layer, used to establish the name scope

      alpha_initializer (None or tf.initializer): Initializer method for alpha array

    """

    ########################################################################
    def __init__(self, name, alpha_initializer=None):
        """
        Constructor
        """
        super(Sinh_JT, self).__init__(name=name)
        with self.name_scope:
            if alpha_initializer is None:
                alpha_initializer = 1.0 * tf.ones(size)
            self.alpha = tf.Variable(alpha_initializer, name="alpha_1")
            self.ea = tf.math.exp(self.alpha)

    ########################################################################
    @tf.Module.with_name_scope
    def forward(self, x):
        """
        Applies the sinh activation and stores the Jacobian

        Args:
            x (tensor): Batch of input vectors

        Returns:
            y (tensor): Output of the DPReLU operation

        """

        with tf.name_scope("forward"):

            self.ea = tf.math.exp(self.alpha)
            y = self.ea * tf.math.sinh(x / self.ea)

            lj = tf.math.log(tf.math.cosh(x / self.ea))
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return y

    ########################################################################
    @tf.Module.with_name_scope
    def inverse(self, y):
        """
        Inverse transformation
        Args:

            y (tensor): Batch of "output" vectors.

        Returns:
            x (tensor): Batch of "input" vectors.

        """

        with tf.name_scope("inverse"):

            self.ea = tf.math.exp(self.alpha)
            x = self.ea * tf.math.asinh(y / self.ea)

            lj = tf.math.log(tf.math.cosh(x / self.ea))
            self.LogJac = tf.math.reduce_sum(lj, axis=1, name="LogJac")

        return x


########################################################################
########################################################################
########################################################################

# class bn_JT(tf.Module):
#     """
#     Batch Normalization
#     """
#
#     ########################################################################
#     def __init__(self, name, eps=1e-7, gamma_initializer=None, beta_initializer=None):
#         """
#         Constructor
#         """
#         super(bn_JT, self).__init__(name=name)
#
#         with self.name_scope:
#             if gamma_initializer is None:
#                 gamma_initializer = 1.0 * tf.ones(1)
#             self.gamma = tf.Variable(gamma_initializer, name="gamma")
#
#             if beta_initializer is None:
#                 beta_initializer = 1.0 * tf.ones(1)
#             self.beta = tf.Variable(beta_initializer, name="gamma")
#
#
#             self.std = 1  # initialize
#             self.mu = 0
#             self.eps = eps
#
#
#
#
#     ########################################################################
#     @tf.Module.with_name_scope
#     def forward(self, x):
#         """
#         Applies the batch normalization and stores the Jacobian
#
#         Args:
#             x (tensor): Batch of input vectors
#
#         Returns:
#             y (tensor): Output of the normalization operation
#
#         """
#
#         with tf.name_scope("forward"):
#             std = tf.math.reduce_std(x, 0)
#             mu = tf.math.reduce_mean(x, 0)
#             std_stable = tf.math.sqrt(std * std + self.eps)
#
#             y = (x - mu) / std_stable
#             diag_scale = 1 / std
#
#             self.LogJac_0 = tf.math.reduce_sum(tf.math.log(diag_scale))
#             self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(y)[0:1], name="LogJac")
#
#         return y
#
#     ########################################################################
#     @tf.Module.with_name_scope
#     def inverse(self, y):
#         """
#         Inverse transformation
#         Args:
#
#             y (tensor): Batch of "output" vectors.
#
#         Returns:
#             x (tensor): Batch of "input" vectors.
#
#         """
#
#         with tf.name_scope("inverse"):
#             self.mean = tf.math.reduce_mean(y, 0)
#             self.std = tf.math.reduce_std(y, 0)
#             self.scale = tf.linalg.diag(1 / self.std)
#             x = (y - self.mean) @ self.scale
#
#             self.LogJac_0 = tf.math.reduce_sum(tf.math.log(diag_scale))
#             self.LogJac = tf.broadcast_to(self.LogJac_0, tf.shape(y)[0:1], name="LogJac")
#
#         return x


########################################################################
########################################################################
########################################################################

class jtprob(tf.Module):
    """
    Basic Jacobian-tracking ANN for density estimation.

    Creates a layer cake of *_JT layers and activations.

    The current forward-transformation log-jacobian is stored in self.LogJac
    each time the forward(), inverse(), or loss() method is invoked.

    The current loss is stored in self.loss_curr every time the loss() method
    is invoked.

    Note that "forward" _always_ means from the input space to the latent space,
    whereas "inverse" always means the reverse i.e. the sampling direction.

    Constructor Args:
          n_in (int): size of input vectors.

          layers (list): List of layers, where each element is a
            Jacobian-tracking FC layer or activation class instance.

          latent_dist (Optional,  a tfp.distributions instance): Distribution
            in latent space to which the distribution in input space is mapped.
            If not specified, the multivariate standard normal distribution is assumed.
            Otherwise it must be a tfp.distributions instance, or at least an instance
            of a class with both log_prob() and sample() methods purporting to represent
            some distribution.  The event_shape property should be n_in.

          aux_diag (None or dict): Dictionary of auxiliary diagnostic functions of latent-space
            variables to be computed during each loss computation. An example that is useful for
            standard normal latent-space distributions is this module's 'Chi2_P' dictionary,
            which can be used to set up reporting of a minibatch's chi^2 and P-value in the latent
            space. The default is None, which means no auxiliary reporting. However, if latent_dist
            is also None, then aux_diag defaults to Chi2_P.

          penalty (None (default) or float): If not None, a penalty term
            will be added -- in training only -- to the loss. The term is
            equal to this value times the variance of the minibatch
            log-Jacobian array.

          name (string): Passed to superclass constructor. Default is "jtprob".

    """

    ########################################################################
    def __init__(self, n_in, layers, aux_diag=None, latent_dist=None,
                 penalty=None, name="jtprob"):
        """
        Constructor

        """
        super(jtprob, self).__init__(name=name)
        setsize(n_in)  # make globally accessible
        if penalty and penalty <= 0.0:
            raise ValueError

        with self.name_scope:
            self.penalty = penalty
            self.layers = layers
            self.aux_diag = aux_diag

            if latent_dist is None:
                self.latent_dist = tfd.MultivariateNormalDiag(loc=np.zeros(n_in, dtype=np.float64))
                if aux_diag is None:
                    self.aux_diag = Chi2_P
            else:
                self.latent_dist = latent_dist

    ########################################################################
    def forward(self, x):
        """
        Forward transformation ( phi(x) )

        The forward log-Jacobian is stored in self.logJac.

        :param x: tf.Tensor; N x size tensor containing vectors in the data space
        :return: tf.Tensor; N x size tensor containing vectors in the latent space
        """

        n_in = getsize()
        with tf.name_scope("forward"):
            # if the rank of x > 2 (e.g x is a set of images) flatten it
            x = tf.reshape(x, (-1, n_in))

            for layer in self.layers:
                x = layer.forward(x)

            self.LogJac = tf.math.add_n([layer.LogJac for layer in self.layers], name="LogJac")
            self.LogJac_curr = tf.math.reduce_mean(self.LogJac, name="Mean_LogJac")

        return x

    ########################################################################
    def inverse(self, y):
        """
        Inverse transformation ( phi^(-1)(y) )

        The forward log-Jacobian is stored in self.logJac.

        :param y: tf.Tensor; N x size tensor containing vectors in the latent space
        :return: tf.Tensor; N x size tensor containing vectors in the data space
        """

        with tf.name_scope("inverse"):
            rlayers = self.layers[:]
            rlayers.reverse()

            for layer in rlayers:
                y = layer.inverse(y)

            self.LogJac = tf.math.add_n([layer.LogJac for layer in self.layers], name="LogJac")
            self.LogJac_curr = tf.math.reduce_mean(self.LogJac, name="Mean_LogJac")

        return y

    ########################################################################
    @tf.Module.with_name_scope
    def loss(self, x):
        """
        Loss function -- negative mean logprob over a minibatch, possibly
        modified by penalty. Also store any auxiliary diagnostics in
        instance attributes with the same name as their corresponding dict
        keys.

        :param x: tf.Tensor; N x size of minibatch of data to compute average negative log likelihood of
        :return: tf.Tensor; tensor containing the real valued loss
        """

        with tf.name_scope("loss") as scope:
            y = self.forward(x)
            self.lp = self.latent_dist.log_prob(y) + self.LogJac
            ls = -tf.math.reduce_mean(self.lp)
            if self.penalty:
                ljvar = tf.math.reduce_var(self.LogJac)
                ls += ljvar * self.penalty
            self.loss_curr = tf.identity(ls, name="current_loss")

            if self.aux_diag is not None:
                for var, fn in self.aux_diag.items():
                    setattr(self, var, fn(y))

        return ls

    ########################################################################
    @tf.Module.with_name_scope
    def sample(self, nsamp):
        """
        Generate a sample from the distribution in latent space and send
        it through the current inverse transformation.

        :param nsamp: int; number of samples to generate
        :return: tf.Tensor; nsamp samples from the distribution defined by the latent dist and transform
        """

        with tf.name_scope("sample") as scope:
            y = self.latent_dist.sample(sample_shape=nsamp)
            x = self.inverse(y)

        return x

########################################################################
########################################################################
########################################################################

class jtprob_TL(tf.Module):
    """
    Class for learning a diffeomorphism where the parameters of the target distribution are simultaneously learned
    (jtprob 'target learning').
    """

    ########################################################################
    def __init__(self, n_in, layers, latent_dist, aux_diag=None,
                 penalty=None, name="jtprob"):
        """
        Constructor

        :param n_in: int; dimension of the data to be modeled
        :param layers: list of tf.Modules; module instances specialized to jacobian tracking (layers)
        :param latent_dist: tfp distribution; containing initial latent dist params
        :param aux_diag: Not currently implemented fully
        :param penalty: float; weight on the penalty
        :param name: string; name the jtprob_TL instance
        """

        super(jtprob_TL, self).__init__(name=name)
        setsize(n_in)  # make globally accessible
        if penalty and penalty <= 0.0:
            raise ValueError

        with self.name_scope:
            self.penalty = penalty
            self.layers = layers
            self.aux_diag = aux_diag

            if latent_dist is None:  # default to MVN with diagonal covariance
                raise ValueError('Please supply a tensorflow distribution instance for the latent distribution.')

            else:
                self.latent_dist = latent_dist
                self.latent_dist_params = [var for var in latent_dist.trainable_variables]
                self.mix = self.latent_dist_params[0]  # mixture params are square root of mixture probabilities


    ########################################################################

    def get_dist_params(self):
        """
        Return a list of the trainable tf variables containing distribution parameters
        :return: ListWrapper of tf.Variables
        """
        return self.latent_dist_params

    ########################################################################
    def forward(self, x):
        """
        Forward transformation ( phi(x) )

        The forward log-Jacobian is stored in self.logJac.

        :param x: tf.Tensor; N x size tensor containing vectors in the data space
        :return: tf.Tensor; N x size tensor containing vectors in the latent space
        """

        n_in = getsize()

        with tf.name_scope("forward"):
            x = tf.reshape(x, (-1, n_in))  # if the rank of x > 2 (e.g x is a set of images) flatten it

            for layer in self.layers:
                x = layer.forward(x)

            self.LogJac = tf.math.add_n([layer.LogJac for layer in self.layers], name="LogJac")
            self.LogJac_curr = tf.math.reduce_mean(self.LogJac, name="Mean_LogJac")

        return x

    ########################################################################
    def inverse(self, y):
        """
        Inverse transformation ( phi^(-1)(y) )

        The forward log-Jacobian is stored in self.logJac.

        :param y: tf.Tensor; N x size tensor containing vectors in the latent space
        :return: tf.Tensor; N x size tensor containing vectors in the data space
        """

        with tf.name_scope("inverse"):
            rlayers = self.layers[:]
            rlayers.reverse()

            for layer in rlayers:
                y = layer.inverse(y)

            self.LogJac = tf.math.add_n([layer.LogJac for layer in self.layers], name="LogJac")
            self.LogJac_curr = tf.math.reduce_mean(self.LogJac, name="Mean_LogJac")

        return y


    ########################################################################
    @tf.Module.with_name_scope
    def loss(self, x):
        """
        Loss function -- negative mean logprob over a minibatch, possibly
        modified by penalty. Also store any auxiliary diagnostics in
        instance attributes with the same name as their corresponding dict
        keys.

        This loss is a modification of the jtprob class loss in that it redefines the target
        (latent) distribution in the forwards pass, which seems to be necessary for training
        distribution parameters in tf 2.0

        :param x: tf.Tensor; N x size of minibatch of data to compute average negative log likelihood of
        :return: tf.Tensor; tensor containing the real valued loss
        """

        with tf.name_scope("loss") as scope:
            y = self.forward(x)

            self._probs_unnormalized = self.mix**2  # self.mix stores unnormalized square root of mixture probabilities
            self.probs = self._probs_unnormalized / tf.reduce_sum(self._probs_unnormalized)

            components = []  # need to update the target distribution manually in tf 2.0
            for i in range(1, len(self.latent_dist_params), 2):
                    components += [tfd.MultivariateNormalDiag(loc=self.latent_dist_params[i],
                                                                   scale_diag=self.latent_dist_params[i+1])]

            self.latent_dist = tfd.Mixture(
                cat=tfd.Categorical(probs=self.probs),
                components=components)

            # self._update_mixture_cdf()

            self.data_term = self.latent_dist.log_prob(y)
            self.lp = self.latent_dist.log_prob(y) + self.LogJac
            ls = -tf.math.reduce_mean(self.lp)

            if self.penalty:
                ljvar = tf.math.reduce_var(self.LogJac)
                ls += ljvar * self.penalty

            self.loss_curr = tf.identity(ls, name="current_loss")

            if self.aux_diag is not None:
                for var, fn in self.aux_diag.items():
                    setattr(self, var, fn(y))

        return ls

    ########################################################################

    @tf.Module.with_name_scope
    def _update_mixture_cdf(self):
        """
        private method for assigning a function to self.latent_dist.cdf, which is not implemented
        for mixtures in tf 2.0. needs to be called in the script or the latent cdf will be out of date

        this is current implemented to handle normal mixtures for latent_dist only
        """

        if True:  # if we are training mix params (fixed mix params case not yet implemented)
            mix = self.latent_dist.trainable_variables[0].numpy()
            mix_probs = mix**2 / np.sum(mix**2)

            scipy_components = []  # list of scipy implementations of the component distributions
            for component_num in range(len(self.latent_dist.components)):
                loc, scale = self.latent_dist.components[component_num].loc, \
                             self.latent_dist.components[component_num].scale

                scipy_components.append(mvn(mean=loc.numpy(), cov=scale.to_dense().numpy()))

        self.latent_dist.cdf = lambda x: np.sum(np.array([mix_probs[i]
                                                          * scipy_components[i].cdf(x)
                                                          for i in range(len(self.latent_dist.components))]))

    ########################################################################

    @tf.Module.with_name_scope
    def sample(self, nsamp):
        """
        Generate a sample from the distribution in latent space and send
        it through the current inverse transformation.

        :param nsamp: int; number of samples to generate
        :return: tf.Tensor; nsamp samples from the distribution defined by the latent dist and transform
        """

        with tf.name_scope("sample") as scope:
            y = self.latent_dist.sample(sample_shape=nsamp)
            x = self.inverse(y)

        return x



########################################################################
########################################################################
########################################################################
class posterior_sampler(jtprob):
    """
    Basic Jacobian-tracking ANN for density estimation.

    Creates a layer cake of *_JT layers and activations.

    The current forward-transformation log-jacobian is stored in self.LogJac
    each time the forward(), inverse(), or loss() method is invoked.

    The loss is computed by sampling from the latent distribution, sending the
    samples through the inverse() transformation, and computing the empirical
    K-L divergence of the transformed distribution from the un-normalized input density.
    The current loss is stored in self.loss_curr every time the loss() method
    is invoked.

    Note that "forward" _always_ means from the input space to the latent space,
    whereas "inverse" always means the reverse i.e. the sampling direction.

    Constructor Args:
          n_in (int): size of input vectors.

          layers (list): List of layers, where each element is a
            Jacobian-tracking FC layer or activation class instance.

          log_input_dens (callable function of input space, returning np.float64): Un-normalized
            log posterior density function to be sampled.

          latent_dist (Optional,  a tfp.distributions instance): Distribution
            in latent space to which the distribution in input space is mapped.
            If not specified, the multivariate standard normal distribution is assumed.
            Otherwise it must be a tfp.distributions instance, or at least an instance
            of a class with both log_prob() and sample() methods purporting to represent
            some distribution.  The event_shape property should be n_in.

          aux_diag (None or dict): Dictionary of auxiliary diagnostic functions of input-space
            variables to be computed during each loss computation. An example that is useful for
            standard normal latent-space distributions is this module's 'Chi2_P' dictionary,
            which can be used to set up reporting of a minibatch's chi^2 and P-value in the latent
            space. The default is None, which means no auxiliary reporting.

          name (string): Passed to superclass constructor. Default is "posterior_sampler".

    """

    ########################################################################
    def __init__(self, n_in, layers, log_input_dens, latent_dist=None, penalty=None,
                 aux_diag=None, name="posterior_sampler"):
        """
        Constructor
        """

        super(posterior_sampler, self).__init__(n_in, layers, aux_diag, latent_dist, penalty, name)

        self.log_input_dens = log_input_dens  # logrho
        print(self.log_input_dens)

    ########################################################################
    @tf.Module.with_name_scope
    def loss(self, nsamp):
        """
        Compute the loss using nsamp samples from the latent distribution.
        """
        with tf.name_scope("loss") as scope:
            y = self.latent_dist.sample(sample_shape=(nsamp))
            x = self.inverse(y)
            ls = self.latent_dist.log_prob(y) + self.LogJac - self.log_input_dens(x)
            #ls = self.LogJac
            ls = tf.reduce_mean(ls)
            self.loss_curr = tf.identity(ls, name="current_loss")

            if self.aux_diag is not None:
                for var, fn in self.aux_diag.items():
                    setattr(self, var, fn(x))

        return ls


########################################################################
########################################################################
########################################################################
#
# Utility functions

def normalize_linear_layer(x, layer):
    """
    Adjust the bias and weight in the Linear_JT layer to match the variances and mean of x

    Args:

      x (ndarray((batchsize, vecsize)): Data -- and _not_ a tf tensor.
      layer (Linear_JT): a Linear_JT fully-connected layer

    The idea is to initialize the layer with a diagonal weight matrix that shrinks
    the axes to unit variance, and initialize the bias vector to mean-subtract the
    data.
    """

    if not isinstance(layer, Linear_JT): raise ValueError

    xbar = np.mean(x, axis=0)
    y = x - xbar
    Cov = y.T.dot(y) / (x.shape[0] - 1)
    siginv = 1.0 / np.sqrt(np.diagonal(Cov))
    Wmat = np.zeros(Cov.shape)
    i = np.diag_indices_from(Wmat)
    Wmat[i] = siginv
    bias = -Wmat.dot(xbar)

    layer.LU = layer.LU.assign(Wmat)
    layer.b = layer.b.assign(bias)


########################################################################
def chisq(y):
    """
    Minibatch Chi^2 of y assuming  y is standard normal
    """

    chi2 = tf.math.reduce_sum(y * y)

    return chi2


########################################################################
def pval(y):
    """
    Minibatch P-Value of y assuming  y is standard normal; by PIT, these should be ~ uniform (apply F_X to x ~ X)
    """

    dof = tf.cast(tf.size(y), dtype=tf.float64)
    dist = tfd.Chi2(dof)
    chi2 = chisq(y)
    pvalue = dist.cdf(chi2)

    return pvalue


########################################################################

Chi2_P = {
    "Chi^2": chisq,
    "P-val": pval
}