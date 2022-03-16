

# TO WORK: NEED TO SET DIM_REDUCE_PATH AND OUTDIR TO NEW DIRS THAT DONT EXIST YET

# script to train and test forecasting machine,
from subprocess import call
import os
import sys
from distutils.dir_util import copy_tree



jobctl_0 = {  # defaults

'seed': 10,
'dim_reduce_directory':
    '/Users/rittlern/Desktop/Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/',
'joint_model_directory':
    '/Users/rittlern/Desktop/Extreme Weather Forecasting/JTrack/JTracker/',
'dim_reduce_outdir':
    '/Users/rittlern/Desktop/Results/',
'joint_model_outdir':
    '/Users/rittlern/Desktop/Results/',


}

sys.path.append('.')  # should set working dir to JTracker first
from Jobs.machine_generator_job import jobctl
jobctl_0.update(jobctl)
globals().update(jobctl)


# require a description of the purpose of this run (what time of year, etc), and require a name for dir to store results
print('\nMake sure you have set job files properly...')
forecaster_description = input('\nInput a description of the forecasting experiment to be run: \n')
forecaster_dir = input('\nName the directory to store results in by test season (e.g. win2020, spr2006): \n')

dim_reduce_outdir += forecaster_dir + '/Dimension Reduction/'
joint_model_outdir += forecaster_dir + '/Joint Model/'


### RUN THE DIMENSION REDUCTION (call read data, then dim reduce)
os.chdir(dim_reduce_directory)

from dim_reduce_job import training_params as dim_reduce_params  # get info on whether to learn or naive dim reduce

call(["python", 'read_data.py'])

if dim_reduce_params['naive']:
    call(["python", 'naive_dim_reduce.py'])
else:
    call(["python", 'dim_reduce.py'])

all_subdirs_dr = [dim_reduce_directory + 'Real_Experiments/' + d for d in
                  os.listdir(dim_reduce_directory + 'Real_Experiments/')
                  if os.path.isdir(dim_reduce_directory + 'Real_Experiments/' + d)]

latest_subdir_dr = max(all_subdirs_dr, key=os.path.getmtime) + '/'  # subdir where the output to dim_reduce stored

### SOME PREP FOR JOINT MODELING
os.chdir(joint_model_directory)

# # get the latest subdir in the folder where models are stored
# all_subdirs_tr = ['Figures/' + d for d in os.listdir('Figures/') if os.path.isdir('Figures/' + d)]
# latest_subdir_tr = max(all_subdirs_tr, key=os.path.getmtime) + '/'
# new_run_number = int(latest_subdir_tr.split('/')[1][8:]) + 1  # assumes the format Wind_Exp'number'
# new_run_dir = joint_model_directory + 'Figures/Wind_Exp' + str(new_run_number) + '/'

from Jobs.tracker_job import jobctl

# if not os.path.exists(new_run_dir):  # if there's a saved job file in the outdir, use that instead
#     os.mkdir(new_run_dir)
# pickle.dump(jobctl, open(new_run_dir + "jobctl", "wb"))

### RUN THE JOINT MODELING
call(['python', 'Driver Scripts/train.py'])
call(['python', 'Verification/testing.py'])  # test the forecast machine on the test data

# add the inputted description of the run to the job file
log_file = jobctl['outdir'] + 'Training_Log'
with open(log_file, 'a+') as lf:
    lf.write('\n')
    lf.write('\nDescription: ' + forecaster_description)


# copy both relevant directories to your results directory
copy_tree(jobctl['dim_reduce_path'], dim_reduce_outdir)  # copy the dim reduce path where the joint training looked to get v_t, so if it lines up with the settings you expect, it is correct
copy_tree(jobctl['outdir'], joint_model_outdir)  # copy the results from looking at that dim reduce path


# seems only potential problem is that you start stacking restults intot the same outdir but you know its always the bottom one that counts if you do stack
# and if you dont see stacking its chill