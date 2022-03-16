

import matplotlib.pyplot as plt
import numpy as np


presentation_directory = '/Users/rittlern/Desktop/Extreme Weather Forecasting/Presentations/'
np.random.seed(10)

# a: sharpness illustration
from scipy.stats import norm

shape_wide = norm(loc=0, scale=10)
# wide_samples = shape_wide.rvs(1000)
wide_grid = np.arange(-40, 40, .1)
wide_pdf = shape_wide.pdf(wide_grid)

shape_sharp = norm(loc=0, scale=1)
sharp_pdf = shape_sharp.pdf(wide_grid)

plt.plot(wide_grid, sharp_pdf, color='orange')
plt.ylabel('Forecast | all accessible info')
plt.title('')
plt.savefig(presentation_directory + 'june/good_info.png')
plt.clf()

plt.plot(wide_grid, wide_pdf, color='orange')
plt.ylabel('Forecast | N/S at Center > -100')
plt.title('')
plt.savefig(presentation_directory + 'june/bad_info.png')
plt.clf()



# a1: well calibration illustration
wc_grid = np.arange(-4, 4, .01)
wc = norm(loc=0, scale=1)
wc_pdf = wc.pdf(wc_grid)
wc_samples = wc.rvs(1000)

fig, ax = plt.subplots()
n, bins, patches = ax.hist(wc_samples, 30, density=1)  # the histogram of the data
ax.plot(wc_grid, wc_pdf, '--')
ax.set_xlabel('N/S Wind, m/s')
ax.set_ylabel('Probability Density of N/S Wind | N/S at Center > -1 ')
ax.set_title('Well-Calibration')
fig.tight_layout()
plt.legend(['Forecast | N/S at Center > -1', 'Empirical Samples | N/S at Center > -1'])
plt.savefig(presentation_directory + 'june/well_calibration.png')
plt.clf()


wc_grid = np.arange(-4, 4, .01)
wc = norm(loc=.5, scale=.35)
wc_pdf = wc.pdf(wc_grid)

fig, ax = plt.subplots()
n, bins, patches = ax.hist(wc_samples, 30, density=1)  # the histogram of the data
ax.plot(wc_grid, wc_pdf, '--')
ax.set_xlabel('N/S Wind, m/s')
ax.set_ylabel('Probability Density of N/S Wind | N/S at Center > -1 ')
ax.set_title('Lack of Well-Calibration')
fig.tight_layout()
plt.legend(['Forecast | N/S at Center > -1', 'Empirical Samples | N/S at Center > -1'])
plt.savefig(presentation_directory + 'june/bad_calibration.png')
plt.clf()


# a2: naive approach

# count=0
# fig, ax = plt.subplots(1, 1, squeeze=False)
# plt.setp(ax, xticks=[], yticks=[])
# response_labels_no_units = [response_labels[i].split(',')[0] for i in range(len(response_labels))]
# predictor_labels_no_units = [predictor_labels[i].split(',')[0] for i in range(len(predictor_labels))]
#
# for p in range(len(predictors)):
#     for lat in range(sq_size):
#         for long in range(sq_size):
#             vec = data[predictors[p]][:, :-lag, lat, long].ravel()
#             full[:, len(responses) + count] = vec
#             corrplot_mats[sq_size-long-1, lat, p] = np.corrcoef(full[:, :1].reshape(len(full[:, :1]),), vec)[1, 0]
#             # T_fake[0, count] = 1 if np.corrcoef(full[:, :1].reshape(len(full[:, :1]),), vec)[1, 0] > .7 else 0
#             count += 1
#
#     if p == 0:
#         ax[0, p].set_title('Correlation of Central N/S Wind, Lag-1 N/S Measurements')
#         im = ax[0, p].imshow(corrplot_mats[:, :, p], interpolation='nearest')
#         pos=fig.add_axes([0.85,0.5,0.02,0.35])
#         fig.colorbar(im, ax=ax[0, p], cax=pos)
# fig.subplots_adjust(right=0.8)
# fig.suptitle('Corr of response and predictors at lag = {}'.format(lag))