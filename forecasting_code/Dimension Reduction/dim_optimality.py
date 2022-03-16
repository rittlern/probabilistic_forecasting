
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
from dim_reduce_v1_functions import path_finder
import os

# get settings
from read_data_job import read_data_params
from dim_reduce_job import training_params
from dim_optimality_job import dim_op_params
from generate_data_job import data_params


# params
globals().update(dim_op_params)
globals().update(read_data_params)
globals().update(training_params)
globals().update(data_params)

sample_size = N if synthetic_data else 22 * (len(range(timerange[0], timerange[1]))-1) * (1-training_percent)

# run read data to read in data of interest
if synthetic_data:
    call(['python', 'generate_data.py'])
else:
    call(['python', 'read_data.py'])

v_losses = []
for dim in range(m_min, m_max, by):
    if synthetic_data:
        path, save_path = path_finder('Experiments/', None)
    else:
        path, save_path = path_finder('Real_Experiments/', None)

    print(path)
    print(save_path)
    # modify the dim_reduce_job.py in order to change the number of dims to reduce to
    with open('dim_reduce_job.py', 'r') as f:
        old = f.read()
    new = old.split("'")
    dim_idx = [i for i in range(len(new)) if 'dimensional target' in new[i]][0]
    new[dim_idx] = new[dim_idx].split(",")[0][:2] + str(dim) + ',' + new[dim_idx].split(",")[1]
    new = "'".join(new)
    with open('dim_reduce_job.py', 'w') as f:
        f.write(new)

    if greedy and dim != m_min:  # modify the dim_reduce_job.py to tell what transform to use in initialization

        # form the previous save path
        prev_path_split = save_path.split('_')
        if prev_path_split[1][0] == '0':
            prev_path_split[1] = '0' + str(int(prev_path_split[1]) - 1) + '/'
        else:
            prev_path_split[1] = str(int(prev_path_split[1][:-1]) - 1) + '/'
        prev_path = '_'.join(prev_path_split)

        new = new.split("'")
        prev_path_idx = [i for i in range(len(new)) if 'model lives' in new[i]][0]
        new[prev_path_idx] = new[prev_path_idx].split(",")[0][:2] + \
                                '"' + prev_path + '"' + ',' + new[prev_path_idx].split(",")[1]

        # find the name of best model in the previous run to send to dim_reduce_job
        prev_Ts = [f for f in os.listdir(prev_path) if f[0] == 'T']
        prev_Ts.sort(key=lambda t: os.path.getmtime(prev_path + t))
        model_name = prev_Ts[0]

        prev_model_idx = [i for i in range(len(new)) if 'to resume training on' in new[i]][0] - 2
        new[prev_model_idx] = new[prev_model_idx].split(",")[0][:2] + \
                              '"' + model_name + '"' + ',' + new[prev_model_idx].split(",")[1]
        new = "'".join(new)

        with open('dim_reduce_job.py', 'w') as f:
            f.write(new)

    # run dim_reduce.py on this dimensional target
    print('here')
    call(['python', 'dim_reduce.py'])

    # get the best validation loss from this run
    v_losses.append(np.min(np.load(save_path + '/validation_losses.npy')))


# refresh the dim_reduce_job.py file so that you don't have to change directories back
if greedy:
    with open('dim_reduce_job.py', 'r') as f:
        old = f.read()
        new = old.split("'")
        save_path_idx = [i for i in range(len(new)) if 'model lives' in new[i]][0]
        new[save_path_idx] = new[save_path_idx].split(",")[0][:2] + 'None' + \
                                  ',' + new[save_path_idx].split(",")[1]
        save_model_idx = [i for i in range(len(new)) if 'to resume training on' in new[i]][0] - 2
        new[save_model_idx] = new[save_model_idx].split(",")[0][:2] + \
                                  'None' + ',' + new[save_model_idx].split(",")[1]
        new = "'".join(new)
    with open('dim_reduce_job.py', 'w') as f:
        f.write(new)

# get estimate of the loss under the identity transformation (under the assumption of Gaussian data)
# loss_full = np.float(np.load(save_path + 'loss_full.npy'))

# get estimate of loss when taking first m PCs
xt = np.load(path + 'x_train.npy')
xt = xt-np.mean(xt, axis=0)
u, s, v = np.linalg.svd(xt, full_matrices=True)

xv = np.transpose(np.load(path + 'x_validation.npy'))
yv = np.load(path + 'y_validation.npy')
xv_proj = np.matmul(v, xv)

y_dim = yv.shape[0]
loss_pca = []
for i in range(m_min, m_max, by):
     Z = np.concatenate((yv, xv_proj[:i, ]), axis=0)
     cov = np.cov(Z)
     loss_pca.append(np.linalg.det(cov[:y_dim, :y_dim] - np.matmul(cov[:y_dim, y_dim:],
                                                      np.linalg.solve(cov[y_dim:, y_dim:], cov[y_dim:, :y_dim]))))

# get estimate of loss when taking first m locations
loss_first = []
for i in range(m_min, m_max, by):
    trans_first = np.zeros([i, xv.shape[0]])
    trans_first[:i, :i] = np.eye(i)
    Z = np.concatenate((yv, np.matmul(trans_first, xv)), axis=0)
    cov = np.cov(Z)
    loss_first.append(np.linalg.det(cov[:y_dim, :y_dim] - np.matmul(cov[:y_dim, y_dim:],
                                                      np.linalg.solve(cov[y_dim:, y_dim:], cov[y_dim:, :y_dim]))))

# plotting
fig, ax = plt.subplots(1, 1)
ax.plot(list(range(m_min, m_max, by)), v_losses)
#ax.plot(list(range(m_min, m_max, by)), loss_first)
#ax.plot(list(range(m_min, m_max, by)), loss_pca)
if not synthetic_data:
    title = 'Reduction Optimality for {} vs. {}; lag = {}; gridsize = {}'.format(responses, predictors, lag, sq_size)
else:
    title = 'Reduction Optimality for Generation Type {}'.format(generation_type)
ax.set_title(title)
ax.set_xlabel('Reduced Dimension')
ax.set_ylabel('Minimum Loss on Validation Set')
if synthetic_data and gaussian_assumption:
    ax.set_yscale('log')
#plt.axhline(y=loss_full, color='r', linestyle='-')
#plt.gca().legend(('Learned Transformation', 'Selection of Individual Locations', 'PCA', 'Without Reduction'))
fig.savefig('Dimension_Optimality/' + title + '.png')
plt.clf()


print('\n')
print('Experiment complete. Results can be found in "Dimension_Optimality/"')
print('\n')

