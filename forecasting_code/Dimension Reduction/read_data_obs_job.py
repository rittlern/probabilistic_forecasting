
read_data_params = {

    # core reading parameters
    'responses': ['vp', 'up'],  # variables to predict; 'up' is wind velocity from west to east, 'vp' is the y wind comp
    'predictors': ['vp', 'up'],  # predictor variables
    'sq_size': 19,  # size of the grid to consider; for now just 19
    'depth': 0,  # how many time steps back to take as predictors in addition to the simulation prediction at time t
    'timerange': [0, 150],  # timesteps of each year to consider; for obs data these are 6 hour intervals
    'yearrange': [0, 21],   # max is year 22 ( which denotes 22 years in the past )

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

