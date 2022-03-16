from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from statsmodels.nonparametric.smoothers_lowess import lowess




# set some stuff up
np.random.seed(0)
latex_font = {'family': 'DejaVu Sans'}


matplotlib.rc('font', **latex_font)


### read in the aggregate data from the runs

with open('/Users/rittlern/Desktop/test_results', 'rb') as f:
    test_data = pickle.load(f)


with open('/Users/rittlern/Desktop/storage683', 'rb') as f:
    storage683 = pickle.load(f)

with open('/Users/rittlern/Desktop/storage954', 'rb') as f:
    storage954 = pickle.load(f)

cal_683_results = np.empty((len(storage683.keys()), len(storage683[list(storage683.keys())[0]])))
for i in range(len(storage683.keys())):
    cal_683_results[i, :] = storage683[list(storage683.keys())[i]]


cal_954_results = np.empty((len(storage954.keys()), len(storage954[list(storage954.keys())[0]])))
for i in range(len(storage954.keys())):
    cal_954_results[i, :] = storage954[list(storage954.keys())[i]]


test_data_array = np.empty((len(test_data.keys()), 1 + len(test_data[list(test_data.keys())[0]])))


# generate keys in correct order
seasons = ['win_', 'spr_', 'sum_', 'fal_']
years = ['01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_']
keys_sorted = []
for year in years:
    for season in seasons:
        keys_sorted.append(season + year + 'dr')
        keys_sorted.append(season + year + 'pca')
        keys_sorted.append(season + year + 'pick')

for i in range(len(keys_sorted)):
    test_data_array[i, :-1] = test_data[keys_sorted[i]]

    if list(keys_sorted)[i][-2:] == 'dr':
        test_data_array[i, 3] = 1

    if list(keys_sorted)[i][-3:] == 'pca':
        test_data_array[i, 3] = 2

    if list(keys_sorted)[i][-4:] == 'pick':
        test_data_array[i, 3] = 3


test_data_array = test_data_array[:108, :]  # if you want to remove the final year of testing to be safe


###################### non-nonparametric fit of all calibration test scores



# # generate some fake data for now
#
# test_scores_954 = np.random.normal(.954, .03, 108).reshape((108, 1))
# test_scores_683 = np.random.normal(.683, .03, 108).reshape((108, 1))

# test_scores = np.concatenate((test_scores_683, test_scores_954), axis=0)
num_bins = 12

fig, ax = plt.subplots(nrows=1, ncols=2)

from scipy import stats
n, bins, patches = ax[0].hist(test_data_array[:, 0], num_bins, density=True,
                              facecolor='green', alpha=0.7, histtype='bar', rwidth=0.9)  #, ec='black')
n, bins, patches = ax[1].hist(test_data_array[:, 1], num_bins, density=True,
                              facecolor='green', alpha=0.7, histtype='bar', rwidth=0.9)  # ec='black')
kde_0 = stats.gaussian_kde(test_data_array[:, 0])
kde_1 = stats.gaussian_kde(test_data_array[:, 1])
x_ticks_0 = np.linspace(.50, .79, 200)
x_ticks_1 = np.linspace(.85, .97, 200)

ax[0].set_ylabel('Count', fontsize=10)
ax[1].set_ylabel('Count', fontsize=10)



ax[0].grid(linestyle='-', linewidth=1, alpha=.25)
ax[1].grid(linestyle='-', linewidth=1, alpha=.25)
ax[0].grid(axis='y', alpha=.3)
ax[1].grid(axis='y', alpha=.3)

ax[0].plot(x_ticks_0, kde_0(x_ticks_0), color='purple')
ax[1].plot(x_ticks_1, kde_1(x_ticks_1), color='purple')


# set xticks
ax[0].set_xticks([.54, .61, .683, .75])
ax[1].set_xticks([.875, .90, .925, .954])

ax[0].set_xticklabels(['.540', '.610', '.683', '.750'])
ax[1].set_xticklabels(['.875', '.900', '.925', '.954'])

mu_hat_683 = np.mean(test_data_array[:, 0])
mu_hat_954 = np.mean(test_data_array[:, 1])
ax[0].text(.53, 9, r'$\hat{\mu}=.670$')
ax[1].text(.87, 16.5, r'$\hat{\mu}=.938$')

# for tick in ax[0].get_yticklabels():
#     tick.set_fontname(latex_font['family'])
#
# for tick in ax[0].get_xticklabels():
#     tick.set_fontname(latex_font['family'])
#
# for tick in ax[1].get_yticklabels():
#     tick.set_fontname(latex_font['family'])
#
# for tick in ax[1].get_xticklabels():
#     tick.set_fontname(latex_font['family'])

plt.subplots_adjust(wspace=.3)

plt.savefig('test_score_hist.png', dpi=500)
plt.clf()


######################  plotting of the validation set calibration scores against iteration for all runs

# generation of some fake data until we get all the runs done (9 years * 4 seasons * 3 runs per season = 108 pairs )
# number_of_tests = 108
# cal_683_mu = .683
# cal_954_mu = .954
# cal_683_results = np.random.normal(cal_683_mu, .04, 108 * 10).reshape((108, 10))  # one
# cal_954_results = np.random.normal(cal_954_mu, .04, 108 * 10).reshape((108, 10))
x_axis = list(range(400, 1300, 100))


###### .683 line points
fig, ax = plt.subplots()
for run in range(108):
    ax.plot(x_axis, cal_683_results[run, :], color='g', alpha=.1)

ax.set_xlabel('Training Iteration', **latex_font)
ax.set_ylabel('Hit Rate over Validation Set, .683 Contour', **latex_font)
ax.grid(alpha=.3, which='major')
ax.grid(alpha=.1, which='minor')
ax.set_ylim(.55, 1.0)
ax.set_xlim(x_axis[0], x_axis[-1])

# set tick labels to correct font, make minor ticks
for tick in ax.get_yticklabels():
    tick.set_fontname(latex_font['fontname'])

for tick in ax.get_xticklabels():
    tick.set_fontname(latex_font['fontname'])

ax.xaxis.set_minor_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(AutoMinorLocator())


plt.savefig('validation_scores1.png')
plt.clf()


### line .954 points  (WANT TO SEE THE HIT RATE GETTING WIDER OVER TIME AS THIS IS EVIDENCE FOR EARLY STOPPING?)

# main plotting code
fig, ax = plt.subplots()
for run in range(108):
    ax.plot(x_axis, cal_954_results[run, :], color='g', alpha=.1)

ax.set_xlabel('Training Iteration', **latex_font)
ax.set_ylabel('Hit Rate over Validation Set, .954 Contour', **latex_font)
ax.grid(alpha=.3, which='major')
ax.grid(alpha=.1, which='minor')
ax.set_ylim(.55, 1.0)
ax.set_xlim(x_axis[0], x_axis[-1])

# set tick labels to correct font, make minor ticks
for tick in ax.get_yticklabels():
    tick.set_fontname(latex_font['fontname'])

for tick in ax.get_xticklabels():
    tick.set_fontname(latex_font['fontname'])

ax.xaxis.set_minor_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(AutoMinorLocator())


plt.savefig('validation_scores2.png')
plt.clf()



###########################  sharpness (plot of all test results of sharpness studies)

# make some fake data for now
# sharpness_dr = np.random.normal(-.05, .1, 108).reshape((108, 1))
# sharpness_pick = np.random.normal(-.25, .1, 108).reshape((108, 1))
# sharpness_pca = np.random.normal(-1, .1, 108).reshape((108, 1))
# sharpness_array = np.concatenate((np.concatenate((sharpness_dr, sharpness_pick), axis=1), sharpness_pca), axis=1)
sharpness_dr = test_data_array[test_data_array[:, 3] == 1, :][:, 2]
sharpness_pca = test_data_array[test_data_array[:, 3] == 2, :][:, 2]
sharpness_pick = test_data_array[test_data_array[:, 3] == 3, :][:, 2]

mu_sharpness_dr = np.mean(test_data_array[test_data_array[:, 3] == 1, :][:, 2])
mu_sharpness_pca = np.mean(test_data_array[test_data_array[:, 3] == 2, :][:, 2])
mu_sharpness_pick = np.mean(test_data_array[test_data_array[:, 3] == 3, :][:, 2])

x_axis = range(len(sharpness_dr))

fit_dr = lowess(sharpness_dr, x_axis)
fit_pca = lowess(sharpness_pca, x_axis)
fit_corr = lowess(sharpness_pick, x_axis)


print(mu_sharpness_dr)
print(mu_sharpness_pca)
print(mu_sharpness_pick)

# basic plotting
x_axis = list(range(36))

fig, ax = plt.subplots()
ax.plot(x_axis, fit_dr[:, 1], 'g-', alpha=.6)
ax.plot(x_axis, fit_corr[:, 1], 'y-', alpha=.6)
ax.plot(x_axis, fit_pca[:, 1], c='purple', alpha=.6)
ax.scatter(x_axis, sharpness_dr, c='green', marker='o', alpha=.4)
ax.scatter(x_axis, sharpness_pick, c='yellow', marker='o', alpha=.4)
ax.scatter(x_axis, sharpness_pca, c='purple', marker='o', alpha=.4)



# tick and label formatting
xticks = ['Winter 1996', 'Winter 1998', 'Winter 2000', 'Winter 2002']
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_xticks(list(range(4, 36, 8)))
ax.set_xticklabels(xticks, **latex_font)
# for tick in ax.xaxis.get_major_ticks()[1::2]:
#     tick.set_pad(15)

ax.yaxis.set_minor_locator(AutoMinorLocator())
for tick in ax.get_yticklabels():
    tick.set_fontname(latex_font['family'])

ax.set_ylabel('Average Test Log-Likelihood (Sharpness)', **latex_font)

# x and y limits
plt.xlim([0, 35])


# add legend
ax.legend(('Mutual Information Maximization', 'Correlation', 'PCA'))


# add grid
ax.grid(which='major', axis='both', alpha=.4)
ax.grid(which='minor', alpha=.15)


plt.savefig('sharpplot', dpi=600)
plt.clf()



######################### hitrate plots (WANT 9 of them for the final paper)






# make table for hyperparam balls, make the figures generated by training pub quality (font changes..),
# architecture graphics, flow chart of training that's a bit more detailed


