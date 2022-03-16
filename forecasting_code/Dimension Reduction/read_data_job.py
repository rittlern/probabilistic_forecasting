

read_data_params = {

    # core reading parameters
    'responses': ['vp', 'up'],  # variables to predict; 'up' is wind velocity from west to east, 'vp' is the y wind comp
    'predictors': ['vp', 'up'],  # predictor variables
    'response_location': [10, 10],  # location in the grid to make the response
    'sq_size': 19,  # size of the grid to consider; for now just 6, 9, 19
    'lag': 1,  # lag*3 = hours between response, predictors
    'depth': 3,  # how many time steps back to take as predictors; 1 means predict 3 hours from now with current weather
    'timerange': [2100, 2900],  # timesteps of each year to consider (Carlo's plots use [0,239])
    'yearrange': [6, 16],   # max is year 22 ( 22 years in the past )  (year 0, 11, 12 are practice test years) (i.e 0 is the first of a ten
    # year interval, the last of which is a practice test set)

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
