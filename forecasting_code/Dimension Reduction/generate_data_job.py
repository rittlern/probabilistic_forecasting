

# generation parameters

data_params = {

    # core generation parameters
    'd': 40,  # original predictor dimension; hard to go above 40 for gentype 2 with current setup without degeneracies
    'N': 1000,  # total sample size
    'generation_type': 2,  # data type to generate

    # secondary generation parameters
    'training_percent': .7,  # percentage data used in the training
    'validation_percent': .1,
    'data_seed': 100,  # seed for data generation

    # I/0 parameters
    'save_run': True,  # should this run be documented?
    'visualize_data': True,  # make plots of generated data
    'max_x_to_plot': 6,  # largest x to include in correlogram if visualizing data


}
