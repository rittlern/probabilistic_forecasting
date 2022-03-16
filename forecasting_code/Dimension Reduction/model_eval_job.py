import numpy as np

model_eval_params = {

    'model_dirs': ['run_58', 'run_59', 'run_61', 'run_62'],  # directories in 'Real Experiments where either cov mats or T live
    'eval_seed': 1,  # seed for the evaluation process
    'size_holdout': 10,  # number of data points to holdout for evaluation
    'number_holdouts': 25,  # number of times to repeat the holdout procedure
    'K': np.array([[50, 60], [60, 90], [80, 90], [100, 120]]),  # rows are knn parameter pairs

}
