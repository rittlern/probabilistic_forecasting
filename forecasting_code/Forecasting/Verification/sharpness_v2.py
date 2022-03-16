


# script for testing the sharpness of estimates relying on a naive KL estimation

import numpy as np
from scipy.spatial import KDTree
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


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
shuffle_idx = np.random.choice(np.array(list(range(test.shape[0]))), size=int(test.shape[0]), replace=False)
test = test[shuffle_idx, :]  # shuffle the indices so that first M examples are not earlier in time than the rest

if M > test.shape[0]:
    M = test.shape[0]

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

grid_logliks = log_densities_paritioned - np.tile(np.reshape(log_marginals, (M, 1)), len(d1) * len(d2))


# # integrate out the predictors so we are left with a density of the climatetology at each point in the test set
# grid_expand = np.tile(grid, (M, 1))
# response_expand = test[:M, :m][np.repeat(np.arange(0, M), len(d1) * len(d2)), :]
# grid_response = tf.constant(np.concatenate((response_expand, grid_expand), axis=1), dtype=tf.float64)
# _ = jtp.loss(grid_response)
# log_densities = jtp.lp
#
# log_densities_paritioned = tf.reshape(log_densities, (-1, len(d1) * len(d2)))  # M rows of grid densities
# log_wind_marginals = tf.math.reduce_logsumexp(log_densities_paritioned + tf.math.log(cell_area), axis=1)  # logprob of wind
#

# naively compute average KL between conditionals and climatetology
_ = jtp.loss(tf.constant(test[:M, :], dtype=tf.float64))
log_test_joint_densities = jtp.lp  # loglik of first M points in the test set
log_test_conditional_densities = tf.reshape(log_test_joint_densities, (1, M)) \
                                 - tf.reshape(log_marginals, (M, 1)).numpy()   # each row is a conditional distribution loglik evaluated at all the test set points
# log_wind_marginals_rep = np.tile(log_wind_marginals.numpy(), (M, 1))
# sharpness = np.mean(log_wind_marginals - log_test_conditional_densities)  # mean of each row is est KL between a conditional and climatetology, and we want mean over these estimates

log_test_liks = np.diagonal(log_test_conditional_densities)  # can be positive too as these are log densities
# np.save(outdir + 'log_test_liks.npy', log_test_liks)

with open(outdir + "sharpness_measures.txt", "w") as file_:
    # print('\nAverage estimated KL divergence between climatetology and conditionals, using {} conditionals: {}'
    #       .format(M, sharpness))
    print('\nAverage test set forecast (conditional) log-likelihood of the model in question, using {} conditionals: {}'
          .format(M, np.mean(log_test_liks)))
    # file_.write('\nAverage estimated KL divergence between climatetology and conditionals, using {} conditionals: {}'
    #             .format(M, sharpness))
    file_.write('\nAverage test set forecast (conditional) log-likelihood of the model in question, '
                'using {} conditionals: {}'.format(M, np.mean(log_test_liks)))


# eventually you could estimate the bias too so that we could see how long we can lag before KL to climate = 0
