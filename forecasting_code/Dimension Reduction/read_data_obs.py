import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
from DimReduceNet import readdata, visualize_training_data


############################################### SETUP #############################################
sys.path.append(".")
np.seterr(divide='raise', over='raise', invalid='raise')

# default params; modify params via read_data_job.py
read_data_params_default = {

    # core reading parameters
    'responses': ['vp', 'up'],  # variables to predict; 'up' is wind velocity from west to east, 'vp' is the y wind comp
    'predictors': ['vp', 'up'],  # predictor variables
    'sq_size': 19,  # size of the grid to consider; for now just 6, 9, 19
    'depth': 0,  # how many time steps back to take as predictors in addition to the simulation prediction at time t
    'timerange': [0, 150],  # timesteps of each year to consider (Carlo's plots use [0,239])
    'yearrange': [0, 20],   # max is year 22 ( which denotes 22 years in the past )

    # secondary read parameters
    'training_percent': .5,
    'validation_percent': .25,  # increasing val is better for model selection
    'double_train': True,  # use separate training sets for dim_reduce and distribution learning
    'yearly_partition': False,  # True -> take all observations of 1 out of every (1/test_percentage) years for test set
    'read_seed': 100,

    # I/0 parameters
    'data_directory_stem': '/Users/rittlern/Desktop/Extreme Weather Forecasting/deep-forecasting/Data/',
    'save_run': True,
    'visualize_data': False,
    'max_x_to_plot': 10,

}


# import the modified reading parameters
try:
    from read_data_wrf_job import read_data_params
    read_data_params_default.update(read_data_params)
except ImportError:
    print("Warning: could not import job configuration -- using defaults")
globals().update(read_data_params_default)

# define the directors to look in for data
data_dir = data_directory_stem + 'ANL_19x19/'

VARIABLE_DESCRIPTIONS = {'vp': 'N/S Wind, m/s', 'up': 'E/W Wind, m/s'}  # blowing south to north in m/s, (w to e too)
response_labels = [VARIABLE_DESCRIPTIONS[string] for string in responses]  # map nc names to understandable names
predictor_labels = [VARIABLE_DESCRIPTIONS[string] for string in predictors]



########################################## READ IN DATA AND FORMAT ###################################

# read in data we want
# BE CAREFUL NO VARIABLES HAVE THE SAME NAME IN MULTIPLE FILES
variables_to_get = list(set(predictors + responses))
data = {}
files = os.listdir(data_dir)
readdata(files, data, variables_to_get, timerange, dir=data_dir)

# some formatting into np arrays
for var, d in data.items():
    dd = np.array(d)  # The indices should now be (year, 3-hr interval, ns, ew)
    print(dd.shape)
    if len(dd.shape) == 5:
        rs = list(dd.shape)
        rs.remove(1)  # Get rid of size 1 dimensions
        dd = dd.reshape(rs)
    data[var] = dd

years = yearrange[1] - yearrange[0]  # number of years
for var in variables_to_get:   # take only the years we want
    data[var] = data[var][yearrange[0]:yearrange[1], :, :, :]


# load in and trim the observational data
obs_data = np.loadtxt(data_directory_stem + 'ANL_obs.txt')
# obs_data[:, 0] = obs_data[:, 0] if > 70 else obs_data[:, 0] + 100
obs_data[obs_data[:, 0] < 70, 0] = obs_data[obs_data[:, 0] < 70, 0] + 100
obs_data[:, 0] = obs_data[:, 0] - obs_data[0, 0]  # get to same year indexing as above
obs_data = obs_data[(obs_data[:, 0] >= yearrange[0]) & (obs_data[:, 0] < yearrange[1])]  # take only the years we want


# create arrays of responses, full data set
step_per_year = len(range(min(timerange), max(timerange)))
response_array = np.empty([years*(step_per_year-depth), len(responses)])  # depth =0 means just use same time wrf preds as prediction, 1 means go 6 hours in the past

for year in range(years):
    yearly_data = obs_data[obs_data[:, 0] == (year + yearrange[0])]
    slice_of_year = yearly_data[(timerange[0] + depth):timerange[1], :]  # can't take first observation if we use observational lag
    print(slice_of_year)
    response_array[(year * (step_per_year - depth)): (year + 1) * (step_per_year - depth), :] = \
        slice_of_year[:, -2:]  #last two columns are repsonses, and also predictors when lagged


# get predictors, and put everything in the full array [ lag is essentially fixed at 1 ]
full = np.empty([response_array.shape[0],
                 (depth+1) * (len(predictors) * sq_size ** 2) + len(responses)])
                    # responses, depth_wrf* wrf predictors, depth * observational/lagged predictors
full[:, :len(predictors)] = response_array


#WORKING FOR 1 YEAR, NOT MULTIPLE YEARS: THIS IS BECAUSE OF MISSING DATA IN YEAR 2. JUST IGNORE MISSING DATA???
response_labels_no_units = [response_labels[i].split(',')[0] for i in range(len(response_labels))]
predictor_labels_no_units = [predictor_labels[i].split(',')[0] for i in range(len(predictor_labels))]
#depth = 1
count = 0
for p in range(len(predictors)):
    for lat in range(sq_size):
        for long in range(sq_size):
            vec = data[predictors[p]][:, depth:, lat, long].ravel()
            full[:, len(responses) + count] = vec
            count += 1

            for dep in range(1, depth + 1):
                vec = data[predictors[p]][:, (depth - dep):-dep, lat, long].ravel()
                full[:, len(responses) + count] = vec
                count += 1


############## HANDLE MISSING VALUES #########################

full[full[:, 0] == 777.70, 0] = .1
full[full[:, 1] == 777.70, 1] = .1  # in the case of calm winds, set the directional speed to .1 (refine later)

full = full[(full[:, 0] < 1000) & (full[:, 1] < 1000), :]  # ignore missing values (assume missing at random)

# check the script with correlation plots after removing 9999s
# corrplot_mats = np.empty([sq_size, sq_size, len(predictors)])
# fig, ax = plt.subplots(1, len(predictors), squeeze=False)
# count = 0
# for p in range(len(predictors)):
#     for lat in range(sq_size):
#         for long in range(sq_size):
#             corrplot_mats[sq_size - long - 1, lat, p] = np.corrcoef(full[:, :1].reshape(len(full[:, :1]), ),
#                                                                     full[:, len(responses) + count])[1, 0]  # a check
#             print(np.corrcoef(full[:, :1].reshape(len(full[:, :1]), ),
#                               full[:, len(responses) + count])[1, 0])
#             count += 1
#
#     ax[0, p].set_title('{} vs. {}'.format(response_labels_no_units[0], predictor_labels_no_units[p], 1))
#     im = ax[0, p].imshow(corrplot_mats[:, :, p], interpolation='nearest')
#     fig.colorbar(im, ax=ax[0, p])
#
# fig.subplots_adjust(right=0.8)
# fig.suptitle('Corr of response and predictors at lag = {}'.format(1))
# fig.show()


############# divide into training, test


recent_test = True
# divide into training, validation, test
if not yearly_partition and not recent_test:  # just divide into training and test uniformly at random, w.o. replacement
    training_idx = np.random.choice(np.array(list(range(0, int(full.shape[0])))),
                                        size=int(full.shape[0]*training_percent), replace=False)
    holdout_idx = [i for i in range(0, int(full.shape[0])) if i not in training_idx]
    validation_idx = np.random.choice(np.array(holdout_idx), size=int(full.shape[0]*validation_percent), replace=False)
    test_idx = [i for i in list(np.array(holdout_idx)) if i not in validation_idx]


if recent_test:  # do not allow any future data to creep into the training or validation (last 3 years all in test)
    # step_per_year = step_per_year-depth
    years = list(range(0, int(full.shape[0] / step_per_year)))
    test_years = [years[0]]  # just take first year in range as test year for now
    print(test_years)

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
# fig.savefig(path + 'corr_plot.png')


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