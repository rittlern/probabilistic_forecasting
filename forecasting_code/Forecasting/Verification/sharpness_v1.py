
# script for testing the sharpness of estimates; uses samples from conditionals and so is more expensive; the KL
# estimation used here should have a lower bias than naive methods stemming from the separate estimation of densities

import numpy as np
from scipy.spatial import KDTree
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


def KL_estimator(s1, s2, k=1):
    """
        credit: https://github.com/nhartland/KL-divergence-estimators

        KL-Divergence estimator using scipy's KDTree
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
    """

    def verify_sample_shapes(s1, s2, k):
        # Expects [N, D]
        assert (len(s1.shape) == len(s2.shape) == 2)
        # Check dimensionality of sample is identical
        assert (s1.shape[1] == s2.shape[1])


    verify_sample_shapes(s1, s2, k)
    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d,  nu_i   = KDTree(s2).query(s1, k)
    rho_d, rhio_i = KDTree(s1).query(s1, k+1)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d/n)*np.sum(np.log(nu_d[::, -1]/rho_d[::, -1]))
    else:
        D += (d/n)*np.sum(np.log(nu_d/rho_d[::, -1]))

    return D


import sys
sys.path.append('.')  # should set working dir to JTracker first
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import numpy as np
import JTNet.JTNet as JTNet


# Default values, changed by tracker_job.py
jobctl_0 = {
#### General config
"logrho" : lambda x: x,             # Log un-nnormalized density
"seed" : np.random.seed,            # Int or None - Seed for tf graph operations and all tf random initialization
#### Model
"vecsize" : 1,                      # Int - Dimension of problems
"layers" : [],
"nstack" : 1,                       # Int - number of layers in a stack
"nfreeze_stacks" : 1,               # Int - number of stacks of layers (how many training phases)
"penalty_const" : None,             # float or None
"latent_dist" : None,               # a tfd instance, 'None' means tfd.MultivariateNormalDiag()
#### Training
"batch_size" : 1,                    # Int - Size of batches, must be less than ntrain
"ntrain" : 1,                       # Int - Size of training data
"ntest" : 1,                        # Int - Size of test data
"nval" : 1,                         # Int - Size of validation data
"max_epoch" : 1,                    # Int - Upper limit of epochs for stopping training
"maxbreak": 1,                      # sets max number of steps with no improvement
"learning_rate" : 0.1,              # learning rate for adam
"optimizer" : None,  # tf.train optimizer instance
#### Checkpointing
"restart" : False,                  # Bool or Int - Either False or path returned by tf.train.Checkpoint.save()
"chkpdir" : "Checkpoints/chkp/",         # string
"tot_min_vloss_oc": 1,              # sets starting point for saving model
#### Logging/plotting
"outdir": "Logging",              # String - Directory where figures, checkpoints, and
"plot_interval": None,              # Int or None - Frequncy of plot
"quiet" : False,                    # Bool
"logfile" : "Training_Log",         # string
"tbdir" : "Tensorboard",          # string
"plot_filename" : "Training.png",   # string
}

from Jobs.tracker_job import jobctl  # eventually get this from the training log
jobctl_0.update(jobctl)
globals().update(jobctl_0)

tf.random.set_seed(seed)
np.random.seed(seed)
tfd = tfp.distributions

# load trained model to test calibration of
jtp = JTNet.jtprob(vecsize, layers, latent_dist=latent_dist, penalty=penalty_const) if not train_params \
    else JTNet.jtprob_TL(vecsize, layers, latent_dist=latent_dist, penalty=penalty_const)

chkp_path = outdir + chkpdir  # 'Figures/Wind_Exp12/Checkpoints/chkp/'  # where the model is stored
chkp = tf.train.Checkpoint(jtp=jtp, Epoch=tf.Variable(1, name='Epoch'), IStack=tf.Variable(0, name='IStack'),
                           minloss=tf.Variable(1e10, name='minloss'))
chkp_manager = tf.train.CheckpointManager(chkp, chkp_path, max_to_keep=3)
chkp.restore(chkp_manager.latest_checkpoint)  # jtp is now the best model from the training (make sure tracker_job set correctly for this)


# load in the test set data
test = np.transpose(np.load(dim_reduce_path + 'v_test.npy'))
test = np.float64(((test - np.mean(test, axis=0)) / np.std(test, axis=0)))  # normalize


# get the likelihoods of each grid point conditioned on each predictor in the test set
cell_length = np.float64(10 ** -precision)
cell_area = cell_length ** 2
d1 = np.arange(-3, 3, cell_length)  # values in direction d1; likewise below for d2
d2 = np.arange(-3, 3, cell_length)
xx, yy = np.meshgrid(d1, d2)
grid = np.float64(np.reshape(np.concatenate((np.reshape(xx, (xx.shape[0], xx.shape[1], 1)),
                                             np.reshape(yy, (yy.shape[0], yy.shape[1], 1))), axis=2),
                                  (len(d1) * len(d2), 2)))

grid_expand = np.tile(grid, (M, 1))
pred_expand = test[:M, m:][np.repeat(np.arange(0, M), len(d1) * len(d2)), :]
grid_pred = tf.constant(np.concatenate((grid_expand, pred_expand), axis=1), dtype=tf.float64)
_ = jtp.loss(grid_pred)
log_densities = jtp.lp  # log densities of the grid points for each predictor in the study

log_densities_paritioned = tf.reshape(log_densities, (-1, len(d1) * len(d2)))  # M rows of grid densities
log_marginals = tf.math.reduce_logsumexp(log_densities_paritioned + tf.math.log(cell_area), axis=1)

grid_logliks = log_densities_paritioned - np.tile(np.reshape(log_marginals, (M, 1)), len(d1) * len(d2))  # compute loglikelihood of each grid point conditional on predictor

max_array = np.max(grid_logliks, axis=1)  # approximate largest log density value for each predictor in test set
max_array_locs = np.argmax(grid_logliks, axis=1)

# for each conditional, generate samples by approximate accept/reject algorithm (approximate as upper bound only approx)
multiple = 20
NN = 150 # number of accept/reject style samples to generate
conditionals_samples = np.empty((M, NN, 2))  # hard code 2 dim response for now
for conditional_idx in range(0, M, 1):

    max_loc = grid[max_array_locs[conditional_idx], :]  # grid is unraveled, this is location of MAP for this pred

    # search over some possible proposals for the one with best approximate acceptance rate
    log_M_opt = sys.float_info.max
    scale_opt = None
    for s in np.arange(.5, 1.0, .001):

        # try two independent Cauchy's
        g = tfd.Cauchy(loc=tf.constant(max_loc, dtype=tf.float64),
                                   scale=tf.constant(s * np.ones(len(max_loc)), dtype=tf.float64))
        g_of_grid = tf.reduce_sum(tf.constant(g.log_prob(grid), dtype=tf.float64), axis=1)
        log_M = np.max((grid_logliks[conditional_idx, :]) - g_of_grid)

        if log_M < log_M_opt:  # if we found a proposal with lower upper bound, keep it
            log_M_opt = log_M
            scale_opt = s

    # define the proposal distribution; center is always approximate conditional mode
    g = tfd.Cauchy(loc=tf.constant(max_loc, dtype=tf.float64),
                   scale=tf.constant(scale_opt * np.ones(len(max_loc)), dtype=tf.float64))
    log_M = log_M_opt  # good news: not hugely dependent on precision (precision = 1 or 2 yield about the same)

    # batch jtp forwards direction
    wind_sample = g._sample_n(multiple*NN)
    pred_expand = np.tile(test[conditional_idx, -m:], multiple * NN).reshape(multiple*NN, 2)
    wind_sample_with_pred_batch = tf.constant(np.concatenate((wind_sample, pred_expand), axis=1), dtype=tf.float64)

    _ = jtp.loss(wind_sample_with_pred_batch)
    log_conditional_density = jtp.lp - log_marginals[conditional_idx]
    log_proposal_density = tf.reduce_sum(g.log_prob(wind_sample), axis=1)

    log_u = np.log(np.random.uniform(size=multiple*NN))

    # need to do this sequentially to guarantee a certain number of samples with probability 1
    sample_idx = np.where(log_u < (log_conditional_density - log_proposal_density - log_M))[0]

    if len(sample_idx) < NN:  #
        raise IndexError('More proposals are needed for conditional number {}. Increase proposal batch (via '
                         'multiple) or run sequentially'.format(conditional_idx))
    else:
        conditionals_samples[conditional_idx, :, :] = wind_sample.numpy()[sample_idx, :][:NN, :]


# extract the climatetology
assert NN < test.shape[0]
climatetology_samples = test[:NN, :-m]  # for now, must use same number as samples taken from conditnls

print(conditionals_samples.shape)

# estimate the KL divergence for each conditional distribution, for a variety of different knn settings
ks = list(range(1, 10, 5))  # how many neighbors to consider in the estimation
KL = np.empty((len(ks), M))

for conditional_idx in range(M):
    print('Estimation for conditional {}....'.format(conditional_idx))
    for ki in range(len(ks)):
        KL[ki, conditional_idx] = KL_estimator(climatetology_samples,
                                                conditionals_samples[conditional_idx, :, :].reshape((NN, 2)), k=ks[ki])

# the final sharpness score is the mean over predictors of estimated KLs
sharpness = np.mean(KL, axis=1)

print(sharpness)

plt.plot(ks, sharpness)
plt.xlabel('k -- as used in the knn estimation of KL divergence')
plt.ylabel('Mean Estimated KL between Climatetology and Forecasts')
plt.savefig(outdir + 'sharpness_estimation.png')

min_k_idx = np.argmax(-sharpness)
with open(outdir + "sharpness_measures.txt", "w") as file_:
    print('\nAverage estimated KL divergence between climatetology and conditionals (over {} predictors, using '
          '{} samples from the conditional distributions): {}'.format(test.shape[0], NN, sharpness[min_k_idx]))
    file_.write('\nAverage estimated KL divergence between climatetology and conditionals (over {} predictors, '
                'using {} samples from the conditional distributions): {}'.format(test.shape[0],
                                                                                  NN, sharpness[min_k_idx]))


# for naive methods, we want an upper bound on KL, as we want to show that our forecasts have higher KL;
#  in general, we want a lower bound on KL as a conservative estimate of our performance