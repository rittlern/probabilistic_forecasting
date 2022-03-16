import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
from DimReduceNet import readdata, visualize_training_data, staircase_plot_simple


############################################### SETUP #############################################
sys.path.append(".")
np.seterr(divide='raise', over='raise', invalid='raise')

# default params; modify params via read_data_job.py
read_data_params_default = {

    # core reading parameters
    'responses': ['up', 'vp'],  # variables to predict
    'predictors': ['vp', 'up'],  # predictor variables
    'response_location': [10, 10],  # location in the grid to make the response
    'sq_size': 19,  # size of the square to consider; for now just 6, 9, 19
    'lag': 1,  # lag*3 = hours between response, predictors
    'timerange': [0, 300],  # timesteps in each year to consider (Carlo's plots use [0,239])

    # secondary read parameters
    'training_percent': .7,
    'read_seed': 1,

    # I/0 parameters
    'data_directory_stem': '/Users/rittlern/Desktop/Extreme Weather Forecasting/deep-forecasting/Data/',
    'save_run': True,
    'visualize_data': True,
    'max_x_to_plot': 10,

}

# import the modified reading parameters
try:
    from read_data_job import read_data_params
    read_data_params_default.update(read_data_params)
except ImportError:
    print("Warning: could not import job configuration -- using defaults")
globals().update(read_data_params_default)

# define the directors to look in for data
subdirs = ['Pressurelevel', 'Surface', 'Wind60m']
data_dirs = [data_directory_stem + 'KC_' + str(sq_size) + 'x' + str(sq_size) + '/' + sub for sub in subdirs]
data_dirs = [di for di in data_dirs if os.path.exists(di)]


#VARIABLE_DESCRIPTIONS = {'vp': 'N/S Wind, m/s', 'up': 'E/W Wind, m/s'}  # blowing south to north in m/s, (w to e too)
VARIABLE_DESCRIPTIONS = {'vp': 'N/S, m/s', 'up': 'E/W, m/s'}  # blowing south to north in m/s, (w to e too)
response_labels = [VARIABLE_DESCRIPTIONS[string] for string in responses]  # map nc names to understandable names
predictor_labels = [VARIABLE_DESCRIPTIONS[string] for string in predictors]

np.random.seed(read_seed)
########################################## READ IN DATA AND FORMAT ###################################

# read in data we want
# BE CAREFUL NO VARIABLES HAVE THE SAME NAME IN MULTIPLE FILES
variables_to_get = list(set(predictors + responses))
data = {}
for dir in data_dirs:
    files = os.listdir(dir)
    readdata(files, data, variables_to_get, timerange, dir=dir)

# some formatting into np arrays
for var, d in data.items():
    dd = np.array(d)  # The indices should now be (year, 3-hr interval, ns, ew)
    if len(dd.shape) == 5:
        rs = list(dd.shape)
        rs.remove(1)  # Get rid of size 1 dimensions
        dd = dd.reshape(rs)
    data[var] = dd

# only save the first yearrange[1] years
years = yearrange[1] - yearrange[0]  # number of years
for var in variables_to_get:
    data[var] = data[var][yearrange[0]:yearrange[1], :, :, :]

# create array of responses
limiter = np.max([lag, depth])  # the number of samples we can take is determined by the larger of lag and depth
step_per_year = len(range(min(timerange), max(timerange)))
response_array = np.empty([years*(step_per_year-(lag+depth-1)), len(responses)])
full = np.empty([response_array.shape[0], depth * (len(predictors) * sq_size ** 2) + len(responses)])
corrplot_mats = np.empty([sq_size, sq_size, len(predictors)])
count = 0

for r in range(len(responses)):
    response_array[:, r] = data[responses[r]][:, (lag+depth-1):,  # increasing indices increases time
                           response_location[0], response_location[1]].ravel()
full[:, :len(responses)] = response_array


fig, ax = plt.subplots(1, len(predictors), squeeze=False)

response_labels_no_units = [response_labels[i].split(',')[0] for i in range(len(response_labels))]
predictor_labels_no_units = [predictor_labels[i].split(',')[0] for i in range(len(predictor_labels))]


for p in range(len(predictors)):
    for lat in range(sq_size):
        for long in range(sq_size):
            for dep in range(depth):
                vec = data[predictors[p]][:, (depth-dep-1):-(lag+dep), lat, long].ravel()
                full[:, len(responses) + count] = vec
                if dep == 0:
                    corrplot_mats[sq_size-long-1, lat, p] = np.corrcoef(full[:, :1].reshape(len(full[:, :1]),),
                                                                        vec)[1, 0]  # a check
                count += 1

    ax[0, p].set_title('{} vs. {}'.format(response_labels_no_units[0], predictor_labels_no_units[p], lag))
    im = ax[0, p].imshow(corrplot_mats[:, :, p], interpolation='nearest')
    fig.colorbar(im, ax=ax[0, p])
fig.subplots_adjust(right=0.8)
fig.suptitle('Corr of response and predictors at lag = {}'.format(lag))


### nicer version for paper
from matplotlib import colors
count = 0
images = []
corrplot_mats_advanced = np.empty([sq_size, sq_size, len(predictors), len(predictors)])
fig, ax = plt.subplots(len(predictors), len(predictors), squeeze=False)
for resp in range(len(predictors)):
    for p in range(len(predictors)):
        for lat in range(sq_size):
            for long in range(sq_size):
                for dep in range(depth):
                    vec = data[predictors[p]][:, (depth-dep-1):-(lag+dep), lat, long].ravel()
                    full[:, len(responses) + count] = vec
                    if dep == 0:
                        corrplot_mats_advanced[sq_size-long-1, lat, resp, p] = np.abs(np.corrcoef(full[:, resp].reshape(len(full[:, resp]),),
                                                                        vec)[1, 0])  # a check
                    count += 1
        print(count)
        ax[resp, p].set_title('{} vs. {}'.format(response_labels_no_units[resp], predictor_labels_no_units[p], lag))
        images.append(ax[resp, p].imshow(corrplot_mats_advanced[:, :, resp, p], interpolation='nearest'))
    count = 0
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

plt.setp(ax, xticks=[2, 16], xticklabels=['$-94.95^{\circ}$', '$-94.15^{\circ}$'],
         yticks=[2, 16], yticklabels=['$39.33^{\circ}$', '$38.81^{\circ}$'])
plt.setp(ax[0,0].get_xticklabels(), fontsize=6)
plt.setp(ax[0,0].get_yticklabels(), fontsize=6)
plt.setp(ax[0,1].get_xticklabels(), fontsize=6)
plt.setp(ax[0,1].get_yticklabels(), fontsize=6)
plt.setp(ax[1,1].get_xticklabels(), fontsize=6)
plt.setp(ax[1,1].get_yticklabels(), fontsize=6)
plt.setp(ax[1,0].get_xticklabels(), fontsize=6)
plt.setp(ax[1,0].get_yticklabels(), fontsize=6)


# fig.subplots_adjust(wspace=.2, hspace=.45)
# fig.colorbar(images[0], ax=ax, orientation='horizontal', pad=.1, fraction=0.046)
# # fig.suptitle('Corr of response and predictors at lag = {}'.format(lag))
# fig.savefig(path + 'corr_plot', dpi=500)




recent_test = True
# divide into training, validation, test
if not yearly_partition and not recent_test:  # just divide into training and test uniformly at random, w.o. replacement
    training_idx = np.random.choice(np.array(list(range(0, int(full.shape[0])))),
                                        size=int(full.shape[0]*training_percent), replace=False)
    holdout_idx = [i for i in range(0, int(full.shape[0])) if i not in training_idx]
    validation_idx = np.random.choice(np.array(holdout_idx), size=int(full.shape[0]*validation_percent), replace=False)
    test_idx = [i for i in list(np.array(holdout_idx)) if i not in validation_idx]


if recent_test:  # do not allow any future data to creep into the training or validation
    step_per_year = step_per_year - (lag + depth - 1)
    years = list(range(0, int(full.shape[0] / step_per_year)))
    test_years = [years[0]]  # take first year in yearrange as test year

    other_years = [year for year in years if year not in test_years]
    other_idx = [list(range(year * step_per_year, (year + 1) * step_per_year)) for year in other_years]
    training_idx = np.random.choice(np.array(other_idx).ravel(),
                                    size=int(len(np.array(other_idx).ravel()) *
                                             training_percent/(training_percent + validation_percent)), replace=False)
    validation_idx = [i for i in np.array(other_idx).ravel() if i not in training_idx]

    test_idx_ = [list(range(year * step_per_year, (year + 1) * step_per_year)) for year in test_years]
    test_idx = np.array([element for list_i in test_idx_ for element in list_i])


if yearly_partition:
    step_per_year = step_per_year - (lag + depth - 1)
    # print(full.shape[0] / step_per_year)  # should be an integer
    years = list(range(0, int(full.shape[0] / step_per_year)))
    training_years = np.random.choice(np.array(years), size=int(len(years) * training_percent), replace=False)
    training_idx_ = [list(range(year * step_per_year, (year + 1) * step_per_year)) for year in training_years]
    training_idx = np.array([element for list_i in training_idx_ for element in list_i])  # just unravel lists

    holdout_years = np.array([year for year in years if year not in training_years])
    validation_years = np.random.choice(np.array(holdout_years), size=int(len(years)*validation_percent), replace=False)
    test_years = [year for year in holdout_years if year not in validation_years]

    validation_idx_ = [list(range(year * step_per_year, (year + 1) * step_per_year)) for year in validation_years]
    validation_idx = np.array([element for list_i in validation_idx_ for element in list_i])
    test_idx_ = [list(range(year * step_per_year, (year + 1) * step_per_year)) for year in test_years]
    test_idx = np.array([element for list_i in test_idx_ for element in list_i])


if double_train:
    training_idx = np.random.permutation(training_idx)
    x_train_dr_idx = training_idx[:int(len(training_idx)/2)]  # dr = dim_reduce
    training_idx = training_idx[int(len(training_idx)/2):int(len(training_idx))]

    x_train_dr = full[x_train_dr_idx, len(responses):]
    y_train_dr = np.transpose(full[x_train_dr_idx, :len(responses)])

    x_train = full[training_idx, len(responses):]
    y_train = np.transpose(full[training_idx, :len(responses)])

else:
    x_train = full[training_idx, len(responses):]
    y_train = np.transpose(full[training_idx, :len(responses)])

print("\nTraining data has {} examples\n".format(x_train.shape[0]))
x_validation = full[validation_idx, len(responses):]
x_test = full[test_idx, len(responses):]
y_validation = np.transpose(full[validation_idx, :len(responses)])
y_test = np.transpose(full[test_idx, :len(responses)])



############################## SAVE SETTINGS. DATA FROM THIS READ IN  ###########################

# create a new directory if necessary
previous_runs = os.listdir('Real_Experiments/')
if len(previous_runs) == 0:
    run_number = 1
else:
    try:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs if s[:3] == 'run']) + 1
    except ValueError:
        print('Place an empty directory called "run_01" in Experiments')
logdir = 'run_%02d' % run_number

path = os.path.join('Real_Experiments', logdir)
if not os.path.isdir(path):
    os.mkdir(path)
path += '/'  # we end all directories with '/' as convention

# save the data in that directory
if double_train:
    np.save(path + 'x_train_dr.npy', x_train_dr)
    np.save(path + 'y_train_dr.npy', y_train_dr)

np.save(path + 'x_train.npy', x_train)
np.save(path + 'y_train.npy', y_train)
np.save(path + 'x_validation.npy', x_validation)
np.save(path + 'y_validation.npy', y_validation)
np.save(path + 'x_test.npy', x_test)
np.save(path + 'y_test.npy', y_test)
fig.savefig(path + 'corr_plot.png')


with open(path + "response_labels.txt", "wb") as rl:  # save the names of the response for plotting later
    pickle.dump(response_labels, rl)

# if the run is to be saved, save the data_generation parameters
if save_run:
    ofd = open(path + 'read_settings.txt', "w")
    ofd.write('##########################\n#\n# Data Reading Parameters:\n')
    for l in read_data_params_default.items():
        ofd.write("#J# " + l[0] + " = " + repr(l[1]) + "\n")
    ofd.close()

############################## VISUALIZATION ############################

if visualize_data:
    visualize_training_data(x_train, y_train, path, max_x_to_plot)

    # joint = np.concatenate((np.transpose(y_train), x_train), axis=1)
    # staircase_plot_simple(joint[:, :10], path + 'training_data_cloud.png', 10, None, 20, ['N/S', 'E/W'],
    #                       ['' for i in range(8)])


#### plot of response for paper (joint, and two marginals)
joint = np.concatenate((np.transpose(y_train), x_train), axis=1)
staircase_plot_simple(joint[:, :10], path + 'training_data_cloud.png', 10, None, 20, ['N/S', 'E/W'],
                           ['' for i in range(8)])