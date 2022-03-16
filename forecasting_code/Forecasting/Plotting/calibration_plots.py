
import numpy as np
import matplotlib.cm as cm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kde
import matplotlib.pyplot as plt
import matplotlib


def hitrate_posterior_plot(hitrate_samples, contours, outdir):
    """

    :param hitrate_samples:
    :param contours:
    :param outdir:
    :return:
    """

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

    nbins = 50
    k = kde.gaussian_kde(np.transpose(hitrate_samples))

    xi, yi = np.mgrid[.65:.72:nbins * 1j, .93:.965:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    map_est_loc = np.vstack([xi.flatten(), yi.flatten()])[:, np.argmax(zi)]  # approx location of MAP

    plt.axvline(x=map_est_loc[0], ls='--', color='0.75')
    plt.axhline(y=map_est_loc[1], ls='--', color='0.75')
    plt.axvline(x=contours[0], ls='--', color='0.75')
    plt.axhline(y=contours[1], ls='--', color='0.75')
    axes.set_title('')
    axes.set_xlabel('Probability of Falling Inside .683 Contour')
    axes.set_ylabel('Probability of Falling Inside .954 Contour')
    axes.grid(True, which='major', axis='both', linestyle='--', color='k', alpha=1)
    im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cm.viridis, alpha=.7)
    cbar = fig.colorbar(im, ax=axes)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Density Value', rotation=270)
    axes.plot(map_est_loc[0], map_est_loc[1],
                        "k^", markersize=15.0, label="Posterior Mode")
    axes.plot(contours[0], contours[1],
                                  "k*", markersize=15.0, label="Calibration Goal")
    axes.legend()
    plt.savefig(outdir + 'hitrate_posterior.png')
    fig.clf()

def conditional_example_plot(xx, yy, d1, d2, grid_logliks, final_conditional_den, contours, clevs,
                             outdir, dim_reduce_path):
    """
    plot an example forecast; # for now: font improve, resolution improve, make 9plot with 1 legend

    :param d1:
    :param d2:
    :param grid_logliks:
    :param final_conditional_den:
    :param contours:
    :param clevs:
    :param outdir:
    :param dim_reduce_path:
    :return:
    """

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')

    mu = np.load(dim_reduce_path + 'response_location.npy')
    sig = np.load(dim_reduce_path + 'response_scale.npy')
    xx, yy = xx * sig[0] + mu[0], yy * sig[1] + mu[1]  # un-normalize the wind values
    gridded = np.reshape(grid_logliks[-1, :], (len(d1), len(d2)))

    # dynamically set figure size
    try:
        hor_args = np.where(gridded > clevs[-1, -1])
        min_hor, max_hor = np.min(hor_args[1]), np.max(hor_args[1])
        min_vert, max_vert = np.min(hor_args[0]), np.max(hor_args[0])

        # cushion = np.min(int((max_hor - min_hor)/10), int((max_vert - min_vert)/10))
        cushion = 7

        min_hor = min_hor - cushion if min_hor > cushion else min_hor  # add a cushion to the plot if we can afford it
        max_hor = max_hor + cushion if max_hor + cushion < gridded.shape[1] else max_hor
        min_vert = min_vert - cushion if min_vert > cushion else min_vert
        max_vert = max_vert + cushion if max_vert + cushion < gridded.shape[0] else max_vert

        gridded_reduction = gridded[min_vert:max_vert, min_hor:max_hor]
        xx_reduction = xx[:, min_hor:max_hor]
        yy_reduction = yy[min_vert:max_vert, :]

    except IndexError:  # in case cushion is too large, no dynamic setting
        print('Cushion too large. Plotting full grid.')
        xx_reduction, yy_reduction, gridded_reduction = xx, yy, gridded

    p = ax.pcolormesh(gridded_reduction, cmap=cm.RdYlGn)  # forecast use cm.RdYlGn, other modeling vidiris

    map = np.unravel_index(np.argmax(gridded_reduction), gridded_reduction.shape)  # location of the MAP
    true = np.unravel_index(np.argmin(np.abs(gridded_reduction - final_conditional_den)), gridded_reduction.shape)

    # adjust ticks to reduction
    # ax.set_xticks(range(0, xx_reduction.shape[1], 5))
    # ax.set_xticklabels(np.round(xx_reduction[0, :-1: 2], decimals=1))
    # ax.set_yticks(range(0, yy_reduction.shape[0], 5))
    # ax.set_yticklabels(np.round(yy_reduction[:-1: 2, 0], decimals=1))
    # ax.tick_params(axis='both', which='major', labelsize=18)

    # cbar = fig.colorbar(p, ax=ax)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('Log Density Value', fontsize=18, rotation=270)

    with open(dim_reduce_path + "response_labels.txt", "rb") as rl:  # get the response names
        response_labels = pickle.load(rl)

    matplotlib.rcParams.update({'font.size': 18})
    p = ax.contour(gridded_reduction, levels=np.flip(clevs[-1, :]), colors="blue")
    p.levels = np.flip(np.array(contours) * 100)
    ax.clabel(p, fontsize=19, fmt="%3.1f%%")

    # unnorm_scale = test[M - 1, 0] * sig[0] + mu[0], test[M-1, 1] * sig[1] + mu[1]  # for checking loc of observation
    # print(unnorm_scale)

    ax.plot(map[1], map[0],  "b^", markersize=20.0, label="Forecast Mode")  # the thing is +2 not applicable in general
    ax.plot(true[1], true[0],
                   "r*", markersize=20.0, label="Delayed Observation")
    ax.legend()
    # ax.set_xlabel(response_labels[0], fontsize=14)
    # ax.set_ylabel(response_labels[1], fontsize=14)
    #ax.set_title("", fontsize=14)

    fig.savefig(outdir + 'conditional_density_example103.png')
    fig.clf()








#
# def pit_plot(pit_estimates_1, pit_estimates_2, dim_reduce_path, outdir):
#
#     n_bins = 15
#     with open(dim_reduce_path + "response_labels.txt", "rb") as rl:
#         response_names = pickle.load(rl)
#
#     for name in range(len(response_names)):
#         if len(response_names[name].split(',')) > 1:
#             response_names[name] = response_names[name].split(',')[0]
#
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#     plt.subplots_adjust(wspace=.4)
#     axes[0].grid(axis='y', alpha=.5)
#     axes[1].grid(axis='y', alpha=.5)
#
#     axes[0].set_ylabel('Density')
#     axes[0].set_xlabel('PIT Estimates for ' + response_names[0])
#
#     axes[1].set_ylabel('Density')
#     axes[1].set_xlabel('PIT Estimates for ' + response_names[1])
#
#     axes[0].hist(pit_estimates_1, n_bins, density=True, histtype='bar', color='blue', label='blue', alpha=.8,
#                  rwidth=0.85)
#     axes[1].hist(pit_estimates_2, n_bins, density=True, histtype='bar', color='blue', label='blue', alpha=.8,
#                  rwidth=0.85)
#
#     plt.savefig(outdir + 'pit.png')
#     plt.clf()
#
#
#
# def multi_conditional_example_plot(xx, yy, d1, d2, grid_logliks, final_conditional_densities, contours, clevs,
#                              outdir, dim_reduce_path):
#
#     """
#     Almost identical to condiitional example plot, but generates 9 plots instead of 1.
#     :param xx:
#     :param yy:
#     :param d1:
#     :param d2:
#     :param grid_logliks:
#     :param final_conditional_densities: a list of conditional densities to plot (with at least 9)
#     :param contours:
#     :param clevs:
#     :param outdir:
#     :param dim_reduce_path:
#     :return:
#     """
#
#     fig = plt.figure(figsize=(70, 30), dpi=400)
#     # fig.set_figwidth(100)
#     # fig.set_figheight(100)
#     fig, axs = plt.subplots(nrows=2, ncols=5)
#
#     density_idx = [0, 1, 2] + [9, 10, 17, 6, 5, 14] + [3]
#     final_conditional_densities = final_conditional_densities[density_idx]  # take the first 9
#
#     mu = np.load(dim_reduce_path + 'response_location.npy')
#     sig = np.load(dim_reduce_path + 'response_scale.npy')
#
#     xx, yy = xx * sig[0] + mu[0], yy * sig[1] + mu[1]  # un-normalize the wind values
#
#     for ax_num in range(len(final_conditional_densities)):
#         ax = axs.reshape(-1)[ax_num]
#         ax.set_aspect('equal', 'box')
#         gridded = np.reshape(grid_logliks[-ax_num, :], (len(d1), len(d2)))
#
#         # dynamically set figure size
#         try:
#             hor_args = np.where(gridded > clevs[-ax_num, -1])
#             min_hor, max_hor = np.min(hor_args[1]), np.max(hor_args[1])
#             min_vert, max_vert = np.min(hor_args[0]), np.max(hor_args[0])
#
#             # cushion = np.min(int((max_hor - min_hor)/10), int((max_vert - min_vert)/10))
#             cushion = 7
#
#             min_hor = min_hor - cushion if min_hor > cushion else min_hor  # add a cushion to the plot if we can afford it
#             max_hor = max_hor + cushion if max_hor + cushion < gridded.shape[1] else max_hor
#             min_vert = min_vert - cushion if min_vert > cushion else min_vert
#             max_vert = max_vert + cushion if max_vert + cushion < gridded.shape[0] else max_vert
#
#             gridded_reduction = gridded[min_vert:max_vert, min_hor:max_hor]
#             xx_reduction = xx[:, min_hor:max_hor]
#             yy_reduction = yy[min_vert:max_vert, :]
#
#         except IndexError:  # in case cushion is too large, no dynamic setting
#             print('Cushion too large. Plotting full grid.')
#             xx_reduction, yy_reduction, gridded_reduction = xx, yy, gridded
#
#         p = ax.pcolormesh(gridded_reduction, cmap=cm.RdYlGn)  # forecast use cm.RdYlGn, other modeling vidiris
#
#         #map = np.unravel_index(np.argmax(gridded_reduction), gridded_reduction.shape)  # location of the MAP
#         true = np.unravel_index(np.argmin(np.abs(gridded_reduction - final_conditional_densities[-ax_num])),
#                                 gridded_reduction.shape)
#
#         # adjust ticks to reduction
#         ax.set_xticks(range(0, xx_reduction.shape[1], 10))
#         ax.set_xticklabels(np.round(xx_reduction[0, :-1: 2], decimals=0))
#         ax.set_yticks(range(0, yy_reduction.shape[0], 5))
#         ax.set_yticklabels(np.round(yy_reduction[:-1: 2, 0], decimals=1))
#
#         if ax_num == len(final_conditional_densities):
#             cbar = fig.colorbar(p, ax=ax)
#             cbar.ax.get_yaxis().labelpad = 15
#             cbar.ax.set_ylabel('Log Density Value', rotation=270)
#             ax.plot(true[1], true[0],
#                     "r*", markersize=15.0, label="Delayed Observation")
#
#         # if ax_num == 1:
#         #     ax.set_ylabel("E/W Wind, m/s", fontsize=14)
#         #
#         # if ax_num == 7:
#         #     ax.set_xlabel("N/S Wind, m/s", fontsize=14)
#
#         p = ax.contour(gridded_reduction, levels=np.flip(clevs[-ax_num, :]), colors="blue")
#         p.levels = np.flip(np.array(contours) * 100)
#         ax.clabel(p, fontsize=14, fmt="%3.1f%%")
#
#         ax.plot(true[1], true[0], "r*", markersize=15, label=None)
#         ax.legend()
#
#
#         ax.set_title("", fontsize=14)
#
#     plt.subplots_adjust(wspace=0.6, hspace=0.2)
#     fig.savefig(outdir + 'conditional_density_example.png')
#     fig.clf()
#
#
# # one N/S m/s and one E/W m/s, one true observation label, one log density thing (foir this, can probably just p]
#






    # add to hyperparam ball, 9 plot, forget raw data, format in the new format; get this done fast as possible so you can work
    # on other things