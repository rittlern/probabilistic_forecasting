import sys
sys.path.append('.')  # should set working dir to JTracker first

import tensorflow as tf
import numpy as np
from scipy.stats import dirichlet

import JTNet.JTNet as JTNet
from Plotting.calibration_plots import conditional_example_plot, hitrate_posterior_plot


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

# load trained model to test calibration of
jtp = JTNet.jtprob(vecsize, layers, latent_dist=latent_dist, penalty=penalty_const) if not train_params \
    else JTNet.jtprob_TL(vecsize, layers, latent_dist=latent_dist, penalty=penalty_const)

chkp_path = outdir + chkpdir  # 'Figures/Wind_Exp12/Checkpoints/chkp/'  # where the model is stored
chkp = tf.train.Checkpoint(jtp=jtp, Epoch=tf.Variable(1, name='Epoch'), IStack=tf.Variable(0, name='IStack'),
                           minloss=tf.Variable(1e10, name='minloss'))
chkp_manager = tf.train.CheckpointManager(chkp, chkp_path, max_to_keep=3)
chkp.restore(chkp_manager.latest_checkpoint)  # jtp is now the best model from the training


# estimate log marginal densities for M predictor tuples in the test set by riemann squares (integration)
# test = np.transpose(np.load(dim_reduce_path + 'v_test.npy'))

print(dim_reduce_path)
test = np.transpose(np.load(dim_reduce_path + 'v_test.npy'))
test = np.float64(((test - np.mean(test, axis=0)) / np.std(test, axis=0)))  # normalize
shuffle_idx = np.random.choice(np.array(list(range(test.shape[0]))), size=int(test.shape[0]), replace=False)
test = test[shuffle_idx, :]  # shuffle the indices so that first M examples are not earlier in time than the rest


if M > test.shape[0]:
    M = test.shape[0]

cell_length = np.float64(10 ** -precision)
cell_area = cell_length ** 2
d1 = np.arange(-3, 3, cell_length)  # values in direction d1; likewise below for d2; (-3,3) good up to 3rd sig fig
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

log_densities_paritioned = tf.reshape(log_densities, (-1, len(d1) * len(d2))).numpy()  # M rows of grid densities
log_marginals = tf.math.reduce_logsumexp(log_densities_paritioned + tf.math.log(cell_area), axis=1).numpy()


# compute log-likelihood of each grid point conditional on predictor
grid_logliks = log_densities_paritioned - np.tile(np.reshape(log_marginals, (M, 1)), len(d1) * len(d2))

# compute the values at which contour levels are attained
descending_logliks = -np.sort(-grid_logliks, axis=1)  # grid densities sorted in descending order by row
contours = [.683, .954]  # where we want contour lines to be drawn
counter = 0
clevs = np.empty((M, len(contours)))  # contours levels for (pred, contour) pairs
cdf = 0
for pred_num in range(M):
    for contour in range(len(contours)):
        while cdf < contours[contour]:
                cdf += np.exp(descending_logliks[pred_num, counter] + np.log(cell_area))
                counter += 1
        clevs[pred_num, contour] = descending_logliks[pred_num, counter-1]
    counter = 0
    cdf = 0

# calibration study; see what percentage of observations fell in respective contours of the conditional distributions
_ = jtp.loss(tf.constant(test[:M, :], dtype=tf.float64))
log_densities_joint = jtp.lp  # joint density of all predictors and true responses
log_densities_conditional = log_densities_joint.numpy() - log_marginals  # conditional density of response given preds
final_conditional_den = log_densities_conditional[-1]  # for plotting example
log_densities_conditional = np.tile(np.reshape(log_densities_conditional,
                                               (len(log_densities_conditional), 1)), len(contours))
hit_rates = np.sum(np.where(log_densities_conditional > clevs, 1, 0), axis=0) / M


# get average conditional log-likelihood on test set (our sharpness measure)
_ = jtp.loss(tf.constant(test[:M, :], dtype=tf.float64))
log_test_joint_densities = jtp.lp  # loglik of first M points in the test set
log_test_conditional_densities = tf.reshape(log_test_joint_densities, (1, M)) \
                                 - tf.reshape(log_marginals, (M, 1)).numpy()   # each row is a conditional distribution loglik evaluated at all the test set points
log_test_liks = np.diagonal(log_test_conditional_densities)  # can be positive too as these are log densities
sharpness_score = np.mean(log_test_liks)
print(sharpness_score)
print(hit_rates)

# write test calibration and sharpness score to log file
with open(outdir + logfile, "a+") as log_fd:
    log_fd.write('\n### Test calibration score (on {} test set samples): {}, {}\n'.format(M, hit_rates[0], hit_rates[1]))
    log_fd.write('\n### Average conditional test likelihood on these examples (sharpness): {}\n'.format(sharpness_score))
