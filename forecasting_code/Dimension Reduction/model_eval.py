
import sys
from dim_reduce_v1_functions import *

# get read parameters
from read_data_job import read_data_params
from model_eval_job import model_eval_params
globals().update(read_data_params)
globals().update(model_eval_params)

# Insert the current directory in the python search path
sys.path.insert(0, ".")

# find the correct paths for loading and saving
stem = 'Real_Experiments/'
path = stem + model_dirs[0] + '/'  # take the data used in that run
#path, _ = path_finder(stem, None)

# load in the data
x_v = np.load(path + 'x_validation.npy')
y_v = np.load(path + 'y_validation.npy')

d = x_v.shape[1]  # the original dimension of the predictor space
p = y_v.shape[0]  # the dimension of the response space (which we use as reduction dimension for now)
m = p  # hard code that the reduced dimension is the sufficient dim reduction dimension for now
n = x_v.shape[0]

# standardize and reshape
x_v = np.transpose((x_v - np.mean(x_v, axis=0)) / np.std(x_v, axis=0))
y_v = np.transpose((np.transpose(y_v) - np.mean(y_v, axis=1)) / np.std(y_v, axis=1))


# for each model, load in the corresponding transformation
np.random.seed(eval_seed)
num_models = len(model_dirs)
Ts = np.empty([m, d, num_models])  # store the transformations here; assume all T have same size for now
for i in range(num_models):
    files = os.listdir('Real_Experiments/' + model_dirs[i])
    T_file = [f for f in files if f[0] == 'T'][-1]  # take file of final transformation in directory if multiple
    Ts[:, :, i] = np.load('Real_Experiments/' + model_dirs[i] + '/' + T_file)

# likelihood comparison
model_scores = np.empty(num_models)
for i in range(num_models):
    grid_scores = np.empty(K.shape[0])
    for point in range(K.shape[0]):
        k_z, k_yz = K[point, 0], K[point, 1]
        likelihoods = np.empty(number_holdouts)
        for j in range(number_holdouts):
            idx_hold = np.random.choice(np.array(range(n-1)),
                                        size=size_holdout, replace=False)  # indices of the holdouts selected at random
            idx_est = np.array([k for k in range(n) if k not in idx_hold])
            y_hold = y_v[:, idx_hold]  # holdout data for likelihood evaluation
            y_est = y_v[:, idx_est]  # data used for estimating the density

            Z = np.matmul(Ts[:, :, i], x_v)
            Z_est = Z[:, idx_est]
            Z_hold = Z[:, idx_hold]

            yZ_est = np.transpose(np.concatenate((y_est, Z_est), axis=0))
            yZ_hold = np.transpose(np.concatenate((y_hold, Z_hold), axis=0))

            V = np.reshape(yZ_hold, (size_holdout, 1, p + m)) - \
                np.reshape(yZ_est, (1, n - size_holdout, p + m))

            Vz = np.prod(V[:, :, p:], axis=2)
            Vy = np.prod(V[:, :, :p], axis=2)
            Vyz = Vz * Vy

            vol_z = np.partition(np.abs(Vz), k_z + 1)[:, k_z]  # for each holdout, volume of parallel to kth est point
            vol_yz = np.partition(np.abs(Vyz), k_yz + 1)[:, k_yz]
            est = (vol_z / vol_yz) * (k_yz / k_z)  # estimated likelihoods of each holdout point

            likelihoods[j] = np.mean(np.log(est))  # average estimated log likelihood of holdout points

        grid_scores[point] = np.mean(likelihoods)  # score for a grid point is average likelihood of holdout points

    model_scores[i] = np.max(grid_scores)  # score for a model is max over the grid search
    print(grid_scores)


print(model_scores)  # note biases may be different as using different Ks in general, so this is not perfect at all
# nice printing of results
