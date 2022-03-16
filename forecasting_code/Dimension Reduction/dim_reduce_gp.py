
###################### hard coded to handle 2d wind data ######################

import sys
from model_functions import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import minimize


# get read parameters
from read_data_job import read_data_params
globals().update(read_data_params)

# Insert the current directory in the python search path
sys.path.insert(0, ".")

# find the correct paths for loading and saving
stem = 'Real_Experiments/'
path, save_path = path_finder(stem, None)

# load in the data
x_t = np.load(path + 'x_train.npy')
y_t = np.load(path + 'y_train.npy')
d = x_t.shape[1]  # the original dimension of the predictor space
p = y_t.shape[0]  # dimension of response

# standardize
x_t = (x_t - np.mean(x_t, axis=0)) / np.std(x_t, axis=0)  # normalize the training set
y_t = (np.transpose(y_t) - np.mean(y_t, axis=1)) / np.std(y_t, axis=1)

# transform data into a friendlier form for the GP computations
v_t = gp_data_reorder(x_t, y_t)

# spatio-temporal coordinate computations
coords_mat = np.empty([np.shape(v_t)[1], 3])  # stores the spatio-temporal coordinates of each observation
for j in range(np.shape(v_t)[1]):
    coords_mat[j, :] = lat_long_t([1, j])

differences = np.reshape(coords_mat[:, :2], (d + p, 1, 2)) - \
              np.reshape(coords_mat[:, :2], (1, d + p, 2))  # pairwise difference in space by coordinate
time_mat = distance_matrix(np.reshape(coords_mat[:, 2], [d + p, 1]),
                           np.reshape(coords_mat[:, 2], [d + p, 1]))  # indicator matrix of time differences


# run the optimization from a few initializations, take params that max likelihood
np.random.seed(1)
num_optim = 1
theta_0 = np.array([1, 0, 5])  # starting with F = (0,0) is agnostic, char = .2 makes smallest kernel value ~ 1/e
batch_size = 4000  # 4604 is max for this data set
# vars = np.var(Y, axis=1)  # good for setting NOISE
NOISE = .6
thetas = np.empty([num_optim, theta_0.shape[0]])
objective_vals = np.empty([num_optim])

for i in range(num_optim):
    print('')
    print('Optimizing on batch {}...'.format(i))
    print('')

    theta_0_ = np.random.normal(theta_0, .001, theta_0.shape[0])
    idx = np.random.randint(0, v_t.shape[0], batch_size)
    Y = v_t[idx, :]
    res = minimize(neg_likelihood_gp, theta_0_, args=(Y, differences, time_mat, NOISE),
                   method='nelder-mead', tol=10)
    thetas[i, :] = res.x
    objective_vals[i] = res.fun

# extract best model of covariance
idx_opt = np.argmin(objective_vals)
theta_opt = thetas[idx_opt, :]

box = np.eye(p)  # learning E/W and N/S independently
tiled = np.tile(box, (int((d+p)/2), int((d+p)/2)))
nugget = NOISE * np.eye(d+p)

f = np.array([theta_opt[1], theta_opt[2]])
F = np.repeat(f, (d+p)**2).reshape([d+p, d+p, 2], order='F')
T = np.repeat(time_mat[:, :, np.newaxis], 2, axis=2)

K_tilde = np.exp(- np.linalg.norm(differences + F * T, axis=2)**2 * theta_opt[0]**2) * tiled + nugget
chol = cho_factor(K_tilde)
amp = np.trace(np.matmul(Y, cho_solve(chol, np.transpose(Y)))) / (Y.shape[0] * Y.shape[1])

cov = amp * K_tilde

##################################### POST TRAINING ###################################################

# save the learned covariance matrix,  visualization of learned covariance matrix, learned transformation
fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0, 0].set_title('Visualization of GP Covariance Matrix')  # cyclic nature makes sense
corr_mat = cov / (cov[0, 0])
im = ax[0, 0].imshow(corr_mat, interpolation='nearest')
fig.colorbar(im, ax=ax[0, 0])

if save_path != path:
    os.mkdir(save_path)

PI = gp_data_reorder_permutation(d)
cov = np.matmul(np.matmul(PI, cov), np.transpose(PI))  # covariance corresponding to original data ordering
np.save(save_path + 'cov.npy', cov)
chol = cho_factor(cov[p:, p:])
np.save(save_path + 'T.npy', np.transpose(cho_solve(chol, cov[p:, :p])))  # normal sufficient dim reduction

print('\n')
print('Training complete. Results can be found in {}'.format(save_path))
print('\n')



# #check that we're actually handling the distances properly
# fig, ax = plt.subplots(1, 1, squeeze=False)
# corr_mat = np.empty((19, 19))
#
# for i in range(0, d-2, 2):
#     coord = lat_long_t([1, 3 + i])
#     corr_mat[int(coord[0]), int(coord[1])] = np.corrcoef(temp_t[:, 1], temp_t[:, 3 + i])[1, 0]
#
# ax[0, 0].set_title('Sanity check for coordinate conversion')
# im = ax[0, 0].imshow(corr_mat, interpolation='nearest')
# fig.colorbar(im, ax=ax[0, 0])
#
# fig, ax = plt.subplots(1, 1, squeeze=False)
# ax[0, 0].set_title('Visualization of Distance Matrix')  #cyclic nature makes sense
# im = ax[0, 0].imshow(dist_mat, interpolation='nearest')
# fig.colorbar(im, ax=ax[0, 0])


