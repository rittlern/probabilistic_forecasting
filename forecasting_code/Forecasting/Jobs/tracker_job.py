import sys
import os
sys.path.insert(0, '../')
sys.path.insert(0, '../JTNet')


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import JTracker.JTNet.JTNet as JTNet


# then clean and comment  train, then work on formatting the output (delete some entries based on style param

# SPECIFY PARAMETERS FOR FORECAST MACHINE TRAINING
jobctl = {
'automated': True,  # parameter telling machine_generator.py to
"seed" : 10,                        # seed for tf and np operations

 # some information on the architecture
'architecture_style': 'coupling',  # options are  'coupling' and 'fully_connected'; see below
'total_params': None,  # total number of trainable parameters in the model (updated below regardless of setting)
"vecsize" : 4,                      # dimension of problem
"layers" : [],  # where layers get stored after architecture constructed below

 # fully connected specific hyperparameters (for controlling layer stacks)
"nlayers" : None,                       # Int - number of layers in a middle stack
"size_bot_stack": 10,                   # number of layers in top stack
"size_top_stack": 1,                    # number of layers in bottom stack
"nfreeze_stacks" : 1,                   # Int - number of stacks of layers )
"num_cycles": 1,  # number of times to run through the training of stacks; e.g. to automate batchsize changes
"restart_stack" : False,                  # Bool or Int - Either False or stack number to restart at when resuming

 # coupling flow specific hyperparameters (mainly for specifying Theta network)
'trans_comps': None,  # list of list of which components get transformed by each successive coupling flow (see below)

"use_resnet": True,  # whether to use a long skip connection linearly transforming input, summing with output (IMPORTANT SOURCE OF COMPLEXITY; CANNOT GET GOOD PERFORMANCE WITHOUT THIS WHILE USING SHALLOW THETAS)
"use_subresiduals": False,  # whether or not to use affine transformed skip connections in intermediate part of Theta
"train_subresiduals": False,  # whether or not to train the subresidual transforms (if False, set to I if possible)
"subresidual_skip_num": 5,  # if using subresiduals, number of 'layers' to skip before reintroducing subresidual
"theta_depth": 7,  # number of affine trans + activation pairs in the Theta network mapping to pseudoparam space


 # main training hyperparameters
"batch_size" : 150,                 # Int - Size of batches, must be less than ntrain
"max_epoch" : 1200,               # Int - Upper limit of epochs for stopping training
#"tot_min_vloss_oc": 100000000000000,  # sets starting point for saving model
"maxbreak": 10000000,  # sets max number of steps with no better model found
"learning_rate" : .01,  # learning rate for adam
'beta_1': .99,  # adam gradient averaging param
'beta_2': .99,  # adam param
"optimizer" : None,  # tf.train optimizer instance, set to adam below
'calibration_interval': 100,  # how often to check validation calibration

 # I/O settings
'dim_reduce_path': '/Users/rittlern/Desktop/'
                   'Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/run_305/',  # in auto mode, needs to be next directory that will be created
"outdir": "./Figures/Wind_Exp305/",   # String - Directory where figures, checkpoints, and
"resume": False,  # whether we restart training on checkpoint in outdir or not

 # misc
"plot_interval": 50,               # Int or None - Frequncy of plot

 # latent distribution
"latent_dist": None,  # distribution to warp into; updated below
'train_params': True,  # train parameters of the latent distribution
"number_mix_components": 5,  # number of mixture components in the latent distribution

 # reguarlization
'p': .4,  # probability with which to accept a new checkpoint in the val loss min checkpointing phase
'minloss_range': [200, 400],  # iterations at which to checkpoint models by new val loss min; after, by val calibration

 # calibration studies
"M": 268,  # number of predictors to use in computation of hit rates, PIT, and sharpness estimation
"m": 2,  # lower dimension of the predictors used
"precision": 1,  # cell length used for probability computations is 10^(precision)
"calibration_plotting": True,  # whether or not to produce plot related to calibration

}

# if automating from dim_reduce through testing, update the dim_reduce_path to the most recent dim_reduce directory
if jobctl['automated']:
    all_subdirs = ['/Users/rittlern/Desktop/'
                   'Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/' + d
                   for d in os.listdir('/Users/rittlern/Desktop/'
                   'Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/')
                   if os.path.isdir('/Users/rittlern/Desktop/'
                   'Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    jobctl['dim_reduce_path'] = latest_subdir + '/'
    print(jobctl['dim_reduce_path'])

# preliminaries for architecture formation/initialization
JTNet.setsize(jobctl["vecsize"])
seed = jobctl["seed"]
tf.random.set_seed(seed)
np.random.seed(seed)  # not overkill; tf and np seeding done separately
jobctl['optimizer'] = tf.optimizers.Adam

# FORMATION OF JTP ARCHITECTURE
if jobctl['architecture_style'] == 'fully_connected':
    nlayers = jobctl["nlayers"]
    nfreeze_stacks = jobctl["nfreeze_stacks"]
    size_top_stack = jobctl["size_top_stack"]
    size_bot_stack = jobctl["size_bot_stack"]

    stack_kern = tf.eye(jobctl["vecsize"], dtype=tf.float64)
    stack_bias = tf.zeros(jobctl["vecsize"], dtype=tf.float64)
    stack_alpha = tf.zeros(jobctl["vecsize"], dtype=tf.float64)

    ki = np.eye(jobctl['vecsize'], dtype=np.float64)

    for ij in range(nfreeze_stacks-1, -1, -1):
        ki = np.eye(jobctl['vecsize'], dtype=np.float64)
        if ij == (nfreeze_stacks - 1) and (nfreeze_stacks > 1):  # the top layer in a multilayer stack smaller usually

            for il in range(size_top_stack):
                jobctl["layers"] += [
                    JTNet.Linear_JT("FC_FREEZE_%3.3d_%3.3d" % (ij, il),
                                    kernel_initializer=ki, initial_bias=stack_bias),
                    JTNet.PReLU_JT("Activation_FREEZE_%3.3d_%3.3d" % (ij, il))]
                print(jobctl['layers'])

        else:
            for il in range(size_bot_stack):
                 jobctl["layers"] += [JTNet.Linear_JT("FC_FREEZE_%3.3d_%3.3d"%(ij, il),
                                                         kernel_initializer=None, initial_bias=stack_bias, seed=seed),
                                      JTNet.PReLU_JT("Activation_FREEZE_%3.3d_%3.3d"%(ij, il))
                                        # , JTNet.Nonlinear_Squared_Coupling(name="Activation_FREEZE_%3.3d_%3.3d"%(ij, il),
                                        #                                    trans_comps=[il%2, il%3])
                                     ]


if jobctl['architecture_style'] == 'coupling':
    use_resnet = jobctl['use_resnet']
    use_subresiduals = jobctl['use_subresiduals']
    train_subresiduals = jobctl['train_subresiduals']
    subresidual_skip_num = jobctl['subresidual_skip_num']
    theta_depth = jobctl['theta_depth']

    jobctl['layers'] += [

        JTNet.Nonlinear_Squared_Coupling(name="Activation_FREEZE_%3.3d_%3.3d" % (0, 0), theta_depth=theta_depth, resnet=use_resnet,
                                         use_subresiduals=use_subresiduals, train_subresiduals=train_subresiduals,
                                        subresidual_skip_num=subresidual_skip_num, seed=seed)
            ,JTNet.Nonlinear_Squared_Coupling(name="Activation_FREEZE_%3.3d_%3.3d"%(0, 1), trans_comps=[2, 3],
                                            theta_depth=theta_depth, resnet=use_resnet,
                                         use_subresiduals=use_subresiduals, train_subresiduals=train_subresiduals,
                                        subresidual_skip_num=subresidual_skip_num, seed=seed)
        , JTNet.Nonlinear_Squared_Coupling(name="Activation_FREEZE_%3.3d_%3.3d" % (0, 2), trans_comps=[1, 2],
                                           theta_depth=theta_depth, resnet=use_resnet,
                                           use_subresiduals=use_subresiduals, train_subresiduals=train_subresiduals,
                                           subresidual_skip_num=subresidual_skip_num, seed=seed)
        # , JTNet.Nonlinear_Squared_Coupling(name="Activation_FREEZE_%3.3d_%3.3d" % (0, 3), trans_comps=[0, 3],
        #                                    theta_depth=theta_depth, resnet=use_resnet,
        #                                    use_subresiduals=use_subresiduals, train_subresiduals=train_subresiduals,
        #                                    subresidual_skip_num=subresidual_skip_num, seed=seed)

         ]

# record which components of the input are transformed by each layer if using coupling flows (not all configs stable)
if len(jobctl['layers']) == 3:
    jobctl['trans_comps'] = [[0, 1], [2, 3], [1, 2]] if jobctl['architecture_style'] == 'coupling' else None
if len(jobctl['layers']) == 2:
    jobctl['trans_comps'] = [[0, 1], [2, 3]] if jobctl['architecture_style'] == 'coupling' else None

# CONSTRUCTION OF LATENT DISTRIBUTION
tfd = tfp.distributions

mix = tf.Variable(np.ones(jobctl["number_mix_components"])/jobctl["number_mix_components"],
                  dtype=tf.float64, trainable=True, name='mix')  # THESE ARE UNNORMALIZED SQUARE ROOTS OF MIX PROBS

components = []
counter = 1
for i in range(1, jobctl['number_mix_components'] + 1):

        loc = tf.Variable(np.random.multivariate_normal(mean=np.zeros(jobctl["vecsize"]),
                                                        cov=np.eye(jobctl["vecsize"])),
                          dtype=tf.float64, trainable=True, name='loc{}'.format(i))
        scale = tf.Variable(np.ones(jobctl["vecsize"]), dtype=tf.float64, trainable=True, name='scale{}'.format(i))
        components += [tfd.MultivariateNormalDiag(loc=loc,
                                                  scale_diag=scale)]
        counter += 1

jobctl['latent_dist'] = tfd.Mixture(cat=tfd.Categorical(probs=mix), components=components)


# compute total trainable parameters in the model
total_parameters = jobctl['number_mix_components'] * 2 * jobctl["vecsize"] + jobctl['number_mix_components']

for layer in jobctl['layers']:  # add in the params from the training of phi
    trainables = layer.trainable_variables  # a tuple of tf.Module instances

    for struct in trainables:
        shape = struct.shape

        shape_params = 1
        for dim in range(len(shape)):
            shape_params *= shape[dim]

        total_parameters += shape_params

jobctl['total_params'] = total_parameters


