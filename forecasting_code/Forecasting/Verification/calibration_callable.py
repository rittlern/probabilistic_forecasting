
# callable version of calibration script with further writing features (to be called during training)
import numpy as np
import tensorflow as tf


def create_grid(precision, M):
    """one time creation of grid used in numerical integration """

    cell_length = np.float64(10 ** -precision)
    cell_area = cell_length ** 2
    d1 = np.arange(-3, 3, cell_length)  # values in direction d1; likewise below for d2
    d2 = np.arange(-3, 3, cell_length)
    xx, yy = np.meshgrid(d1, d2)
    grid = np.float64(np.reshape(np.concatenate((np.reshape(xx, (xx.shape[0], xx.shape[1], 1)),
                                                 np.reshape(yy, (yy.shape[0], yy.shape[1], 1))), axis=2),
                                 (len(d1) * len(d2), 2)))

    grid_expand = np.tile(grid, (M, 1))

    return grid_expand, cell_area, d1, d2


def calibration_check(jtp, dim_reduce_path, grid_expand, cell_area, d1, d2, M, m, outdir, seed=10):
    """ check calibration of a model on the validation set """

    tf.random.set_seed(seed)
    np.random.seed(seed)

    val = np.transpose(np.load(dim_reduce_path + 'v_v.npy'))  # hard coded to get calibration on validation set
    val = np.float64(((val - np.mean(val, axis=0)) / np.std(val, axis=0)))  # normalize
    shuffle_idx = np.random.choice(np.array(list(range(val.shape[0]))), size=int(val.shape[0]), replace=False)
    val = val[shuffle_idx, :]  # legacy: shuffle indices so that first M examples aren't earlier in time than the rest

    if M > val.shape[0]:
        M = val.shape[0]

    pred_expand = val[:M, m:][np.repeat(np.arange(0, M), len(d1) * len(d2)), :]
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
            clevs[pred_num, contour] = descending_logliks[pred_num, counter - 1]
        counter = 0
        cdf = 0

    # calibration study; see what percentage of observations fell in respective contours of conditional distributions
    _ = jtp.loss(tf.constant(val[:M, :], dtype=tf.float64))
    log_densities_joint = jtp.lp  # joint density of all predictors and true responses
    log_densities_conditional = log_densities_joint.numpy() - log_marginals  # conditional density of response | preds
    log_densities_conditional = np.tile(np.reshape(log_densities_conditional,
                                                   (len(log_densities_conditional), 1)), len(contours))
    hit_rates = np.sum(np.where(log_densities_conditional > clevs, 1, 0), axis=0) / M

    # # do some writing of the results to files
    # with open(outdir + "calibration_measures.txt", "w") as file_:
    #     print('\nHit Rates by Contour (number of hits over {} val set samples):'.format(M))
    #     file_.write('Hit Rates by Contour (number of hits over {} val set samples):'.format(M))
    #     for contour in range(len(contours)):
    #         print('{} contour: {}'.format(contours[contour], hit_rates[contour]))
    #         file_.write('\n{} contour: {}\n'.format(contours[contour], hit_rates[contour]))

    return hit_rates
