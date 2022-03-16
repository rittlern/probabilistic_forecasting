import sys
sys.path.append('.')  # should set working dir to JTracker first
from time import gmtime, strftime, time
import os
import sys
import re
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib

import JTNet.JTNet as JTNet
from Plotting.tracker_plots_bw import tracker_plots
# from Plotting.staircase import staircase_plot_simple
import pickle
from Verification.calibration_callable import *


tf.executing_eagerly()

tstart = time()
tstart_str = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime(tstart))

sys.path += "[.]"
# rs.maintain_state("tracker_rngstate")
np.seterr(divide="raise", over="raise", invalid="raise")

############## Job Control ##################################
# Default values, changed by tracker_job.py
jobctl_0 = {
"seed" : 10,                        # seed for tf and np operations

 # some information on the architecture
'architecture_style': 'coupling',
'total_params': None,
"vecsize" : 4,
"layers" : [],

 # fully connected specific hyperparameters (for controlling layer stacks)
"nlayers" : None,
"size_bot_stack": 10,
"size_top_stack": 1,
"nfreeze_stacks" : 1,
"num_cycles": 1,
"restart_stack" : False,

 # coupling flow specific hyperparameters (for specifying Theta network)
"use_resnet": True,
"use_subresiduals": False,
"train_subresiduals": False,
"subresidual_skip_num": 5,
"theta_depth": 7,


 # main training hyperparameters
"batch_size" : 250,
"max_epoch" : 550,
"maxbreak": 10000000,
"learning_rate" : .01,
'beta_1': .99,
'beta_2': .99,
"optimizer" : None,

 # I/O settings
'dim_reduce_path': '/Users/rittlern/Desktop/'
                   'Extreme Weather Forecasting/deep-forecasting/Code/Dimension Reduction/Real_Experiments/run_211/',
"outdir": "./Figures/Wind_Exp132/",
"resume": False,
"logfile" : "Training_Log",
"tbdir" : "Tensorboard",
"plot_filename" : "training_plot.png",
"chkpdir": "Checkpoints/chkp/",

 # misc
"plot_interval": 50,

 # latent distribution
"latent_dist": None,
'train_params': True,
"number_mix_components": 5,

 # reguarlization
'p': .4,
'minloss_range': [200, 400],

 # calibration studies
"M": 1000,
"m": 2,
"precision": 1,
"calibration_plotting": True,

}

from Jobs.tracker_job import jobctl
jobctl_0.update(jobctl)
globals().update(jobctl_0)

#
# # some stuff for machine_generator
# all_subdirs_tr = ['Figures/' + d for d in os.listdir('Figures/') if os.path.isdir('Figures/' + d)]
# latest_subdir_tr = max(all_subdirs_tr, key=os.path.getmtime) + '/'
# # new_run_number = int(latest_subdir_tr.split('/')[1][8:]) + 1  # assumes the format Wind_Exp'number'
# # new_run_dir = 'Figures/Wind_Exp' + str(new_run_number) + '/'
# print(latest_subdir_tr)
#
# if os.path.isfile('./' + latest_subdir_tr + "jobctl"):  # if there's a saved job file in the outdir, use that instead
#     jobctl = pickle.load(open('./' + latest_subdir_tr + "jobctl", 'rb'))
#     jobctl_0.update(jobctl)
#     globals().update(jobctl_0)
#
# print(outdir)
tf.random.set_seed(seed)
np.random.seed(seed)  # not overkill; tf and np seeding done separately

################## Output Control ###################################
epoch_oc = []
trloss_oc = [] ; trljac_oc = []
vloss_oc = [] ; vljac_oc = []

outctl = [["Epoch", "epoch_oc", epoch_oc],
          ["Batch-Size Training Loss", "trloss_oc", trloss_oc],
          ["Batch-size Validation Loss", "vloss_oc", vloss_oc],
          ["Batch-Size Training Log-Jacobian" , "trljac_oc", trljac_oc],
          ["Batch-size Validation Log-Jacobian" , "vljac_oc", vljac_oc]]

ocdict={}
for out in outctl:
    ocdict[out[1]] = out[2]

calibration_scores = {}

# remove some unnecessary parts of dict so that log file is easier to understand
if architecture_style == 'coupling':
    jobctl.pop('nlayers', None)
    jobctl.pop('size_bot_stack', None)
    jobctl.pop('size_top_stack', None)
    jobctl.pop('nfreeze_stacks', None)
    jobctl.pop('restart_stack', None)

if architecture_style == 'fully_connected':
    jobctl.pop('trans_comps', None)
    jobctl.pop('use_resnet', None)
    jobctl.pop("use_subresiduals", None)
    jobctl.pop("train_subresiduals", None)
    jobctl.pop("subresidual_skip_num", None)
    jobctl.pop("theta_depth", None)

##########################
# Some I/O
##########################

try:
    os.stat(outdir)
except OSError:
    os.mkdir(outdir)

log_fd = open(outdir + "/" + logfile, "a+")
log_fd.write('##########################\n#\n# Jacobian Tracker Training Output\n#\n')
log_fd.write('##########################\n#\n# %s\n' % tstart_str)
log_fd.write('##########################\n#\n# Job input:\n')
for l in jobctl.items():
    log_fd.write("#J# " + l[0] + " = " + repr(l[1]) +"\n")


log_fd.write('#\n# Columns:\n')
for c in range(len(outctl)):
    log_fd.write('# %2.2d = %s\n' % (c, outctl[c][0]))

for filename in os.listdir(outdir):
    if filename[:3] == 'inv':
        os.remove(outdir + filename)


print(dim_reduce_path)
with open(dim_reduce_path + "response_labels.txt", "rb") as rl:
    response_names = pickle.load(rl)
response_names = [response_names[i].split(',')[0] for i in range(len(response_names))]
pred_names = [['$T(x)_{1}$', '$T(x)_{2}$'], ['$T(x)_{1}$', '$T(x)_{2}$', '$T(x)_{3}$'],
              ['$T(x)_{1}$', '$T(x)_{2}$', '$T(x)_{3}$', '$T(x)_{4}$']][int(vecsize - len(response_names) - 2)]


# load in the transformed wind data
v_t = np.transpose(np.load(dim_reduce_path + 'v_t.npy'))  # training data
v_v = np.transpose(np.load(dim_reduce_path + 'v_v.npy'))  # validation data

# normalize (the response has already been normalized, but not the 'artificial predictors')
train = np.float64((v_t - np.mean(v_t, axis=0)) / np.std(v_t, axis=0))
val = np.float64((v_v - np.mean(v_v, axis=0)) / np.std(v_v, axis=0))

ntrain = train.shape[0]
nval = val.shape[0]

grid_expand, cell_area, d1, d2 = create_grid(precision, M)  # one-time creation of grid for numerical integration

# staircase_plot_simple(train, outdir + 'training_data_cloud.png', vecsize, None, 20, response_names, pred_names)
# staircase_plot_simple(val, outdir + 'val_data_cloud.png', vecsize, None, 20)

#####################
# Initialize JTNN
#####################

jtp = JTNet.jtprob(vecsize, layers, latent_dist=latent_dist, penalty=None) if not train_params \
    else JTNet.jtprob_TL(vecsize, layers, latent_dist=latent_dist, penalty=None)

##########################
# Some checkpoint management
########################
chkp_path = outdir + chkpdir  # we always end paths with '/'
chkp = tf.train.Checkpoint(jtp=jtp, Epoch=tf.Variable(1, name='Epoch'), IStack=tf.Variable(0, name='IStack'),
                           best_model_val_loss=tf.Variable(1e10, name='minloss'),
                           hit_rate_683=tf.Variable(1e10, name='hr683'),
                           hit_rate_954=tf.Variable(1e10, name='hr954'))

chkp_manager = tf.train.CheckpointManager(chkp, chkp_path, max_to_keep=1)
if resume is False:
    best_model_val_loss_global, delta_global = sys.float_info.max, sys.float_info.max
    epoch_0 = 0
    istack_0 = 0
else:
    chkp.restore(chkp_manager.latest_checkpoint)
    epoch_0 = chkp.Epoch.numpy()
    istack_0 = chkp.IStack.numpy()
    best_model_val_loss = chkp.best_model_val_loss.numpy()
    hit_rate_683, hit_rate_954 = chkp.hit_rate_683.numpy(), chkp.hit_rate_954.numpy()
    delta_global = hit_rate_683 + hit_rate_954
    best_epoch = epoch_0  # epoch that the minloss_global attained at


##########################
# Set up Logging
########################

tb_writer = tf.summary.create_file_writer(outdir + tbdir)
tb_writer.set_as_default()

##########################
# Showtime
########################
count = 0  # for early stopping in the case of no improvment
plot = 0  # for plotting of intermediate models

max_epoch = nfreeze_stacks * [max_epoch] if isinstance(max_epoch, int) else max_epoch
learning_rate = nfreeze_stacks * [learning_rate] if isinstance(learning_rate, float) else learning_rate
batch_size = nfreeze_stacks * [batch_size] if isinstance(batch_size, int) else batch_size

for cycle in range(num_cycles):
    for istack in range(istack_0, nfreeze_stacks):
        chkp.IStack.assign(istack)

        max_epoch_stack = max_epoch[istack]
        learning_rate_stack = learning_rate[istack]
        batch_size_stack = batch_size[istack]

        opz = optimizer(learning_rate=learning_rate_stack, beta_1=beta_1, beta_2=beta_2)  # redefine opz to change lr

        # Get the trainable variables for this layer
        tv = []
        for layer in layers:

            # CURRENTLY, FINAL CYCLE TRAINS EVERY TRAINABLE VARIABLE
            if cycle == (num_cycles - 1) and (num_cycles > 1):  # train everything in the last cycle
                tv += list(layer.trainable_variables)

            else:
                nm = layer.name_scope.name[:-1]  # this is where tf.Module stashes the name it is passed
                ist = int(nm.split(sep="_")[-2])  # this is the stack number embedded in the name
                if ist != istack:
                    continue
                tv += list(layer.trainable_variables)

        if train_params:
            tv += jtp.get_dist_params()

        for epoch in range(chkp.Epoch.numpy() + 1, chkp.Epoch.numpy() + max_epoch_stack + 1):
            with tf.GradientTape(watch_accessed_variables=True) as tape:

                tr_batch_idx = np.random.randint(0, ntrain, batch_size_stack)
                tr_batch = tf.constant(train[tr_batch_idx, :], dtype=tf.float64)
                tr_batch_loss = jtp.loss(tr_batch)

                tr_loss = jtp.loss(tf.constant(train, dtype=tf.float64))

            val_batch_idx = np.random.randint(0, nval, nval)
            val_batch = tf.constant(val[val_batch_idx, :], dtype=tf.float64)
            val_batch_loss = jtp.loss(val_batch)

            trloss_oc.append(tr_loss.numpy())
            trljac_oc.append(jtp.LogJac_curr.numpy())
            tf.summary.scalar("trloss", tr_batch_loss, step=epoch)
            tf.summary.scalar("trljac", jtp.LogJac_curr.numpy(), step=epoch)

            vloss_oc.append(val_batch_loss.numpy())
            vljac_oc.append(jtp.LogJac_curr.numpy())
            tf.summary.scalar("vloss", val_batch_loss, step=epoch)
            tf.summary.scalar("vljac", jtp.LogJac_curr.numpy(), step=epoch)

            tb_writer.flush()
            epoch_oc.append(epoch)

            grad = tape.gradient(tr_batch_loss, tv)
            opz.apply_gradients(zip(grad, tv))

            # record what we've done
            if epoch % 25 == 0:
                print('Iteration: {:4.0f};  Batch Training Loss: {:0.7f};  Batch Val Loss: {:0.7f}'.
                      format(epoch, tr_batch_loss, val_batch_loss))
                # if train_params:
                #     print(jtp.get_dist_params())  # watch the dynamics of the latent space
                #     print('')

            ostr = "%6.6d " % epoch
            for out in outctl[1:]:
                exec("buf = "+out[1]+"[-1]")
                ostr += "%11.5E " % (buf)
            log_fd.write("%s\n" % ostr)
            log_fd.flush()

            if epoch % plot_interval == 0:
                tracker_plots(ocdict, outdir + plot_filename, plotpenalty=False)

            # # checkpointing conditions
            # if val_batch_loss.numpy() < best_model_val_loss_global and epoch >= minloss_range[0]:
            #
            #     if np.random.binomial(1, p) > 0:  # we only take this as a minimum with prob p to regularize
            #         best_model_val_loss_global = val_batch_loss.numpy()
            #         chkp.best_model_val_loss.assign(best_model_val_loss_global)
            #         best_epoch = epoch
            #         chkp_manager.save()
            #
            #         # print('\nNew validation min attained at iteration {}: {}\n'.format(epoch,
            #         #                                                                    best_model_val_loss_global))
            #
            #         print('Iteration: {:4.0f};  New Validation Min Attained....  Batch Val Loss: {:0.7f}'.
            #               format(epoch, val_batch_loss))



            # if epoch % calibration_interval == 0 and epoch >= minloss_range[1]:
            #         hit_rates = calibration_check(jtp, dim_reduce_path, grid_expand, cell_area, d1, d2, M, m, outdir)
            #         delta_current = 1.3 * np.abs(hit_rates[0] - .683) + np.abs(
            #         hit_rates[1] - .954)  # weighted sum of l1 dist to target
            #         calibration_scores[str(epoch)] = hit_rates
            #
            #         print('\n' + ' ' * 12 + 'Calibration score attained: {}, {}\n'.format(hit_rates[0],
            #                                                                                     hit_rates[1]))
            #
            #         if delta_current < .05:
            #             chkp_manager.save()
            #             best_epoch = epoch
            #             break

            if epoch % calibration_interval == 0 and epoch >= minloss_range[1]:  # after a while, switch to calibration for saving
                hit_rates = calibration_check(jtp, dim_reduce_path, grid_expand, cell_area, d1, d2, M, m, outdir)
                delta_current = 1.3 * np.abs(hit_rates[0] - .683) + np.abs(hit_rates[1] - .954)  # weighted sum of l1 dist to target
                calibration_scores[str(epoch)] = hit_rates

                print(delta_current)
                print(delta_global)
                print(delta_current < delta_global)

                if delta_current < delta_global:
                    if np.random.binomial(1, 1):
                        best_model_val_loss_global = val_batch_loss.numpy()  # save val loss even under new criterion
                        chkp.best_model_val_loss.assign(best_model_val_loss_global)
                        chkp.hit_rate_683.assign(hit_rates[0])
                        chkp.hit_rate_954.assign(hit_rates[1])
                        chkp_manager.save()

                        delta_global = delta_current
                        best_epoch = epoch

                        print('\n' + ' '*12 + 'New Calibration min attained: {}, {}\n'.format(hit_rates[0], hit_rates[1]))


            else:
                count += 1
                if count > maxbreak:
                    break

            chkp.Epoch.assign_add(1)

        chkp.restore(chkp_manager.latest_checkpoint).assert_consumed()  # weights don't update til data is fed to model

# some diagnostic plots, records updates

print('\nBest model found at epoch {}\n'.format(best_epoch))
print('\nMin validation loss attained at this epoch {}\n'.format(best_model_val_loss_global))
print('\nPlotting Post Training Diagnostics...')

yy = jtp.forward(tf.constant(val, dtype=tf.float64))  # this is when params get updated to best model
tar = jtp.latent_dist
y = tar.sample(10000).numpy()
x = jtp.inverse(tf.constant(y, dtype=tf.float64))

#print(np.sum(np.isnan(x.numpy())))  # useful to diagnose degeneracy in inversion
staircase_plot_simple(x.numpy(), outdir + 'inverse_staircase.png', vecsize, None, 20, response_names, pred_names)
staircase_plot_simple(yy.numpy(), outdir + 'forward_staircase.png', vecsize, None, 20)


log_fd = open(outdir + "/" + logfile, "a+")
log_fd.write('#############\n#\n#############')
log_fd.write('#\n# Validation Calibration Scores:\n')
for iter in calibration_scores.keys():
    log_fd.write("#J# " + 'Iter ' + iter + ': ' + str(calibration_scores[iter][0]) + ', '
                 + str(calibration_scores[iter][1]) +"\n")
log_fd.write('\n### Best validation calibration achieved at iteration {}\n'.format(best_epoch))
log_fd.write('#############')
log_fd.close()

# for i in range(5):
#     os.system(" say training done ")


