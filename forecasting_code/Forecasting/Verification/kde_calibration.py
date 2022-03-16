import sys
sys.path.append('.')  # should set working dir to JTracker

# fits a KDE model and then estimates calibration metrics of the KDE model

# the idea here is that a necessary condition for bad calibration coming from something other than joint modeling
# is bad calibration here

# it feels like modeling the problem as for different, simpler data we had well calibration

from sklearn.neighbors import KernelDensity
import sklearn
import numpy as np
import tensorflow as tf


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


from Jobs.tracker_job import jobctl
jobctl_0.update(jobctl)
globals().update(jobctl_0)

np.random.seed(seed)

### load in the data

v_t = np.transpose(np.load(dim_reduce_path + 'v_t.npy'))
v_v = np.transpose(np.load(dim_reduce_path + 'v_v.npy'))
v_t = np.concatenate((v_t, v_v), axis=0)
test = np.transpose(np.load(dim_reduce_path + 'v_test.npy'))
shuffle_idx = np.random.choice(np.array(list(range(test.shape[0]))), size=int(test.shape[0]), replace=False)
test = test[shuffle_idx, :]  # shuffle the indices so that first M examples are not earlier in time than the rest


# normalize (the response has already been normalized, but not the 'artificial predictors')
train = np.float64((v_t - np.mean(v_t, axis=0)) / np.std(v_t, axis=0))
test = np.float64((test - np.mean(test, axis=0)) / np.std(test, axis=0))

### fit a KDE to the training data

params = {'bandwidth': np.arange(.01, .4, .001)}
grid = sklearn.model_selection.GridSearchCV(KernelDensity(), params)
grid.fit(train)

kde = grid.best_estimator_


###  just as in calibration script, integrate out the response to get marginals
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
grid_pred = np.concatenate((grid_expand, pred_expand), axis=1)
log_densities = tf.constant(kde.score_samples(grid_pred), dtype=tf.float64)

log_densities_paritioned = tf.reshape(log_densities, (-1, len(d1) * len(d2))) # M rows of grid densities
log_marginals = tf.math.reduce_logsumexp(log_densities_paritioned + tf.math.log(cell_area), axis=1).numpy()


# compute log-likelihood of each grid point conditional on predictor
grid_logliks = log_densities_paritioned.numpy() - np.tile(np.reshape(log_marginals, (M, 1)), len(d1) * len(d2))


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
log_densities_joint = kde.score_samples(test[:M, :])
log_densities_conditional = log_densities_joint - log_marginals  # conditional density of response given preds
final_conditional_den = log_densities_conditional[-1]
log_densities_conditional = np.tile(np.reshape(log_densities_conditional,
                                               (len(log_densities_conditional), 1)), len(contours))
hit_rates = np.sum(np.where(log_densities_conditional > clevs, 1, 0), axis=0) / M
print(log_densities_conditional)
print(np.sum(np.isnan(log_densities_conditional)))
print(hit_rates)



