import sys
sys.path.append('.')  # should set working dir to JTracker first

import tensorflow as tf
import numpy as np
from scipy.stats import dirichlet

import JTNet.JTNet as JTNet
from Plotting.calibration_plots import conditional_example_plot


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

chkp_path = outdir + chkpdir  #'Figures/Wind_Exp297/Checkpoints/chkp/'  # where the model is stored  #
chkp = tf.train.Checkpoint(jtp=jtp, Epoch=tf.Variable(1, name='Epoch'), IStack=tf.Variable(0, name='IStack'),
                           minloss=tf.Variable(1e10, name='minloss'))
chkp_manager = tf.train.CheckpointManager(chkp, chkp_path, max_to_keep=3)
chkp.restore(chkp_manager.latest_checkpoint)  # jtp is now the best model from the training


# estimate log marginal densities for M predictor tuples in the test set by riemann squares (integration)
# test = np.transpose(np.load(dim_reduce_path + 'v_test.npy'))

test = np.transpose(np.load(dim_reduce_path + 'v_test.npy'))
test = np.float64(((test - np.mean(test, axis=0)) / np.std(test, axis=0)))  # normalize
#test_y = np.load(dim_reduce_path + 'y_test.npy')
shuffle_idx = np.random.choice(np.array(list(range(test.shape[0]))), size=int(test.shape[0]), replace=False)
test = test[shuffle_idx, :]  # shuffle the indices so that first M examples are not earlier in time than the rest


if M > test.shape[0]:
    M = test.shape[0]


# you vary M you can get different 'last examples' to plot
# this is nice for a merge https://pinetools.com/merge-images
precision = 1.15
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

final_conditional_densities = log_densities_conditional[-20:]
final_conditional_den = log_densities_conditional[-1]  # for plotting example
true_response = test[-1, :2]
log_densities_conditional = np.tile(np.reshape(log_densities_conditional,
                                               (len(log_densities_conditional), 1)), len(contours))
hit_rates = np.sum(np.where(log_densities_conditional > clevs, 1, 0), axis=0) / M
# print(log_densities_conditional)
# print(np.sum(np.isnan(log_densities_conditional)))

# # do a bayesian estimation of the hitrates; done jointly
# log_densities_conditional_0, log_densities_conditional_1 = log_densities_conditional[:, 0], \
#                                                            log_densities_conditional[:, 1]
# x_11 = np.sum(np.where((log_densities_conditional_0 > clevs[:, 0]) & (log_densities_conditional_1 > clevs[:, 1]), 1, 0))
# x_00 = np.sum(np.where((log_densities_conditional_0 < clevs[:, 0]) & (log_densities_conditional_1 < clevs[:, 1]), 1, 0))
# x_01 = np.sum(np.where((log_densities_conditional_0 < clevs[:, 0]) & (log_densities_conditional_1 > clevs[:, 1]), 1, 0))
# x_10 = np.sum(np.where((log_densities_conditional_0 > clevs[:, 0]) & (log_densities_conditional_1 < clevs[:, 1]), 1, 0))
#
# prior_ALPHA = np.ones(4)  # if all prior alpha are ones, we get an uninformative dirichlet prior
# post_alpha = np.array([prior_ALPHA[0] + x_11, prior_ALPHA[1] + x_00, prior_ALPHA[2] + x_01, prior_ALPHA[3] + x_10])
#
# post_samples = dirichlet.rvs(post_alpha, size=1000, random_state=seed)
# hitrate_samples = np.empty((1000, 2))
# hitrate_samples[:, 0] = post_samples[:, 0] + post_samples[:, 3]  # post samples over Pr(true sample in first contour)
# hitrate_samples[:, 1] = post_samples[:, 0] + post_samples[:, 2]  # post samples over Pr(true sample in 2nd contour)


# do some writing of the results to files
with open(outdir + "calibration_measures.txt", "w") as file_:
    print('\nHit Rates by Contour (number of hits over {} test set samples):'.format(M))
    file_.write('Hit Rates by Contour (number of hits over {} test set samples):'.format(M))
    for contour in range(len(contours)):
        print('{} contour: {}'.format(contours[contour], hit_rates[contour]))
        file_.write('\n{} contour: {}\n'.format(contours[contour], hit_rates[contour]))

# plotting of posterior distribution over hitrates
# hitrate_posterior_plot(hitrate_samples, contours, outdir)

# plotting of conditional densities
conditional_example_plot(xx, yy, d1, d2, grid_logliks, final_conditional_den, contours, clevs,
                         outdir, dim_reduce_path)

# # plot 9 conditional densities
# multi_conditional_example_plot(xx, yy, d1, d2, grid_logliks, final_conditional_densities, contours, clevs,
#                              outdir, dim_reduce_path)



# get these 9 plots looking good,
# exapand hyperparam table
#  start nice formatting ( put into new template )