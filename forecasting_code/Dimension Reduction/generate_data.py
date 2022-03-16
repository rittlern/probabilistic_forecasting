import numpy as np
import os
from dim_reduce_v1_functions import visualize_training_data
import matplotlib.pyplot as plt

############################## JOB CONTROL ##########################################

# default data parameters, changed by generate_data_job.py
data_params_default = {

    # core generation parameters
    'd': 40,  # original predictor dimension; hard to go above 40 for gentype 11 with current setup without degeneracies
    'N': 10000,  # total sample size
    'generation_type': 1,  # data type to generate

    # secondary generation parameters
    'training_percent': .7,  # percentage data used in the training vs. validation
    'data_seed': 10,  # seed for data generation

    # I/0 parameters
    'save_run': True,  # should this run be documented?
    'visualize_data': False,  # make plots of generated data
    'max_x_to_plot': 6,  # largest x to include in correlogram if visualizing data

}

try:
    from generate_data_job import data_params
    data_params_default.update(data_params)
except ImportError:
    print("Warning: could not import job configuration -- using defaults")
globals().update(data_params_default)


################################ DATA GENERATION  ####################################
np.random.seed(data_seed)

if generation_type == 1:
    Sig = np.zeros(shape=[d+1, d+1])
    assert float(d) % 2 == 0, 'Please specify d divisible by 2 for gen_type 1'

    yx_dim = int(d/5)
    Sig_xx = np.eye(int(d/5)) * 100
    Sig_xx = np.matmul(np.transpose(Sig_xx), Sig_xx)
    Sig_zz = np.random.normal(0, 1, size=[int(4*yx_dim), int(4*yx_dim)])
    Sig_zz = np.matmul(np.transpose(Sig_zz), Sig_zz)
    Sig_yx = np.full(yx_dim, 70)

    Sig[0, 0] = 10
    Sig[1:yx_dim+1, 1:yx_dim+1] = Sig_xx
    Sig[yx_dim+1:, yx_dim+1:] = Sig_zz
    Sig[1:yx_dim+1, 0] = Sig_yx
    Sig[0, 1:yx_dim+1] = Sig_yx

    cov = Sig
    assert np.all(np.linalg.eigvals(Sig) > 0), 'Covariance Matrix is not positive definite; cannot generate normal data'

    mu = np.full(d + 1, 1)
    data = np.random.multivariate_normal(mu, Sig, N)
    x_data = data[:, 1:]
    y_data = data[:, 0].reshape(data[:, 0].shape[0], 1)

    # T_opt = np.transpose(np.linalg.solve(Sig[1:, 1:], Sig[1:, :1]))
    # yz = np.transpose(np.concatenate((np.transpose(y_data), np.matmul(T_opt, np.transpose(x_data))), axis=0))
    # print(yz)
    # print(np.mean(yz[:,0]))
    # print(np.mean(yz[:, 1]))

    # visualize the information gain by adding predictors one by one
    # if visualize_data:
    #     res = []
    #     p = 1
    #     for i in range(1, 100):
    #         trans_first_i = np.zeros([i+1, d+1])
    #         trans_first_i[:(i+1), :(i+1)] = np.eye(i+1)
    #         cov = np.matmul(np.matmul(trans_first_i, Sig), np.transpose(trans_first_i))
    #         loss = np.linalg.det(cov[:p, :p] - np.matmul(cov[:p, p:],
    #                                                                     np.linalg.solve(cov[p:, p:],
    #                                                                                     cov[p:, :p])))
    #         res.append(loss)
    #
    #     plt.plot(res)
    #     plt.show()
    #     plt.clf()

elif generation_type == 2:
    # a 2D response version with only some useful predictors
    Sig = np.zeros(shape=[d + 2, d + 2])
    assert float(d) % 2 == 0, "please specify d divisible by 2 for gen_type 2"

    yx_dim = int(d / 5)
    Sig_xx = np.eye(int(d / 5)) * 100
    Sig_xx = np.matmul(np.transpose(Sig_xx), Sig_xx)
    Sig_zz = np.random.normal(0, 2, size=[int(4 * yx_dim), int(4 * yx_dim)])
    Sig_zz = np.matmul(np.transpose(Sig_zz), Sig_zz)
    Sig_yx = np.concatenate((np.full(yx_dim, 70).reshape(1, yx_dim), np.full(yx_dim, 60).reshape(1, yx_dim)))

    Sig[0, 0] = 10
    Sig[1, 1] = 10
    Sig[2:yx_dim + 2, 2:yx_dim + 2] = Sig_xx
    Sig[yx_dim + 2:, yx_dim + 2:] = Sig_zz
    Sig[2:yx_dim + 2, :2] = np.transpose(Sig_yx)
    Sig[:2, 2:yx_dim + 2] = Sig_yx

    cov = Sig
    assert np.all(np.linalg.eigvals(Sig) > 0), "Covariance matrix is not positive definite; can't generate normal data"

    mu = np.full(d + 2, 1)
    data = np.random.multivariate_normal(mu, Sig, N)
    x_data = data[:, 2:]
    y_data = data[:, :2].reshape(data[:, 0].shape[0], 2)

else:
    means = np.linspace(-2, 2, d)
    gamma = np.random.poisson(5, size=d)/2

    if generation_type == 3:
        # an almost block diagonal covariance structure
        if d % 4 != 0:
            raise Exception("Input data dimension should be divisible by four for gen_type 3")

        half = int(d/2)
        quarter = int(d/4)

        cov1 = np.random.normal(3.5, 2, size=[half, half])
        cov1 = np.matmul(np.transpose(cov1), cov1)

        cov2 = np.random.normal(3, 2, size=[quarter, quarter])
        cov2 = np.matmul(np.transpose(cov2), cov2)

        cov3 = np.random.normal(1, 1, size=[quarter, quarter])
        cov3 = np.matmul(np.transpose(cov3), cov3)

        cov = np.random.normal(0, 1, size=[d, d])
        cov = np.matmul(np.transpose(cov), cov)

        cov[:half, :half] += cov1
        cov[half:(half + quarter), half:(half + quarter)] += cov2
        cov[(half + quarter):d, (half + quarter):d] += cov3

        gamma = np.empty(d)
        gamma[:half] = 1
        gamma[half:(half + quarter)] = 2
        gamma[(half + quarter):d] = .01

    if generation_type == 4:
        # x ~ multivariate t distribution in this case
        cov = np.random.normal(2, 2, size=[d, d]) * np.power(-1, np.random.binomial(n=1, p=.7, size=[d, d]))
        cov = np.matmul(np.transpose(cov), cov)

        dof = 1
        u = np.random.gamma(dof/2., 2./dof, size=(N, 1))
        w = np.random.multivariate_normal(np.zeros(means.shape[0]), cov, size=N)

    if generation_type == 5:
        # another 2d response normal case
        cov = np.random.normal(2, 2, size=[d, d]) * np.power(-1, np.random.binomial(n=1, p=.7, size=[d, d]))
        cov = np.matmul(np.transpose(cov), cov)
        gamma1 = np.random.normal(loc=0, scale=1, size=d)
        gamma2 = np.random.normal(loc=0, scale=1, size=d)

    # generate predictors
    if generation_type == 4:
        x_data = w / np.tile(np.sqrt(u), [1, means.shape[0]]) + means
    else:
        x_data = np.random.multivariate_normal(means, cov, size=N)

    # generate responses
    if generation_type == 5:
        y_data = np.empty((N, 2))
        for i in range(N):
            y_data[i, :] = np.random.multivariate_normal([np.dot(x_data[i, :], gamma1), np.dot(x_data[i, :], gamma2)],
                                                         np.eye(2), size=1)  # still independent noise
    else:
        y_data = np.empty((N, 1))
        for i in range(N):
            y_data[i, :] = np.random.normal(np.dot(x_data[i, :], gamma), 1, size=1)  # still independent noise


# divide into training, validation, test ( note that this is overkill as samples generated iid; copied code here )
training_idx = np.random.choice(np.array(list(range(0, int(x_data.shape[0])))),
                                    size=int(x_data.shape[0]*training_percent), replace=False)
holdout_idx = [i for i in range(0, int(x_data.shape[0])) if i not in training_idx]
validation_idx = np.random.choice(np.array(holdout_idx), size=int(x_data.shape[0]*validation_percent), replace=False)
test_idx = [i for i in list(np.array(holdout_idx)) if i not in validation_idx]

x_train = x_data[training_idx, :]
x_validation = x_data[validation_idx, :]
x_test = x_data[test_idx, :]

y_data = np.transpose(y_data)
y_train = y_data[:, training_idx]
y_validation = y_data[:, validation_idx]
y_test = y_data[:, test_idx]


############################ SAVE SETTINGS. DATA FROM THIS GENERATION ###########################

# create a new directory if necessary
previous_runs = os.listdir('Experiments/')
if len(previous_runs) == 1:
    run_number = 1
else:
    try:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs if s[:3] == 'run']) + 1
    except ValueError:
        print('Place an empty directory called "run_01" in Experiments')
logdir = 'run_%02d' % run_number

path = os.path.join('Experiments', logdir)
if not os.path.isdir(path):
    os.mkdir(path)

# save the data in that directory
np.save(path + '/x_train.npy', x_train)
np.save(path + '/y_train.npy', y_train)
np.save(path + '/x_validation.npy', x_validation)
np.save(path + '/y_validation.npy', y_validation)
np.save(path + '/x_test.npy', x_test)
np.save(path + '/y_test.npy', y_test)

if generation_type != 1 and generation_type != 2:
    if generation_type != 5:
        np.save(path + '/gamma.npy', gamma)
    else:
        np.save(path + '/gamma_1.npy', gamma1)
        np.save(path + '/gamma_2.npy', gamma2)

np.save(path + '/cov.npy', cov)
if 'mu' in globals():
    np.save(path + '/mu.npy', mu)

# if the run is to be saved, save the data_generation parameters
if save_run:
    ofd = open(path + '/data_settings.txt', "w")
    ofd.write('##########################\n#\n# Data Generation Parameters:\n')
    for l in data_params_default.items():
        ofd.write("#J# " + l[0] + " = " + repr(l[1]) + "\n")
    ofd.close()

############################### MAKE SOME CORRELATION PLOTS, SAVE THEM #####################

if visualize_data:
    visualize_training_data(x_train, y_train, path, max_x_to_plot)



