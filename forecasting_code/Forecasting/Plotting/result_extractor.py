# script to get results from the logging files and aggregate them in one place

import os
import pickle

desktop = '/Users/rittlern/Desktop/'  # main path
ancestor_dir = desktop + 'Results/'


#### get the validation calibration scores

storage683 = {}
storage954 = {}
tests = {}

milestones = ['Iter 400', 'Iter 500', 'Iter 600', 'Iter 700', 'Iter 800',
              'Iter 900', 'Iter 1000', 'Iter 1100', 'Iter 1200']

for year_dir in os.listdir(ancestor_dir):
    if year_dir[:4] == 'Year':  # in this case, it's truly a year's result storage folder
        for run_dir in os.listdir(ancestor_dir + year_dir):

            if run_dir[:3] in ['win', 'spr', 'sum', 'fal']:
                print(run_dir)
                training_log = ancestor_dir + year_dir + '/' + run_dir + '/Joint Model/Training_Log'

                with open(training_log, 'r') as log:
                    data = log.read()

                    results683, results954, testing = list(), list(), list()
                    for iter in range(len(milestones)):
                        start_pos = data.find(milestones[iter])

                        if iter != 8:
                            end_pos = data.find('#J#', start_pos, -1)  # this is the position of the next '#J#'
                        else:
                            end_pos = data.find('###', start_pos, -1)

                        slice_of_interest = data[start_pos:end_pos]
                        contents = slice_of_interest.split(':')[1].split(',')
                        results683.append(float(contents[0]))
                        results954.append(float(contents[1]))

                    storage683[run_dir] = results683
                    storage954[run_dir] = results954

                    # get the test data too
                    test_calibration_start = data.find('test set samples')
                    test_calibration_end = data.find('###', test_calibration_start, -1)

                    slice_of_interest = data[test_calibration_start:test_calibration_end]
                    contents_test_cal = slice_of_interest.split(':')[1].split(',')
                    testing.append(float(contents_test_cal[0]))
                    testing.append(float(contents_test_cal[1]))

                    sharpness_start = data.find('sharpness')
                    sharpness_end = data.find('Description')
                    sharpness_slice = data[sharpness_start:sharpness_end]
                    testing.append(float(sharpness_slice.split(':')[1]))
                    tests[run_dir] = testing





# save the results to file
with open(desktop + 'storage683', 'wb') as f:
    pickle.dump(storage683, f)

with open(desktop + 'storage954', 'wb') as f:
    pickle.dump(storage954, f)

with open(desktop + 'test_results', 'wb') as f:
    pickle.dump(tests, f)

