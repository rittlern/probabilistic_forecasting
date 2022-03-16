#!/usr/bin/python3

import sys
from importlib import import_module
import numpy as np
import scipy.optimize as op
import tensorflow as tf
import JTNet
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from sklearn.preprocessing import  StandardScaler

sys.path.insert(0,".")
np.seterr(divide="raise", over="raise", invalid="raise")

############## Job Control ##################################
#
# Defaults, changed by FWV_job.py
#
jobctl_0 = {
        "tf_chk_path": None,
        "tracker_jobctl": None,
        "data_path": None,
        "norm_filename": None,
        "val": 5101,
        "nsamp": 40,
        "contours" : [0.683, 0.954],
        "cont_plot_stem": "pcontour_",
        "speed_plot_stem": "wspeed_",
        "axis_labels": ["E-W Wind Vel. (m s$^{-1}$)", "N-S Wind Vel. (m s$^{-1}$)"],
        "simplex_scale": 0.5,
        "xdomain" : (-7.0,4.0),
        "ydomain" : (-10.0, 1.0),
        "ll_min" : None,
        "smax": 15.0,
        "s_samp": 100,
        "cplot_title": "3-Hour Delay Wind Velocity Forecast:\n Colormap is Log Probability Density",
        "splot_title": "3-Hour Delay Wind Speed Forecast",
}

from FWV_job import jobctl
jobctl_0.update(jobctl)
globals().update(jobctl_0)

jt = import_module(tracker_jobctl)
globals().update(jt.jobctl)

################################################################

with open(norm_filename,'rb') as f:
    scaler = pickle.load(f)

##########################
# Some data management
##########################
def readnpfile(data_file):
    data = np.load(data_file)
    if hasattr(data, "files"): # .npz file, assume first "file" is the data
        return data[data.files[0]]
        data.close()
    else:
        return data

datavectors = readnpfile(data_file)
vecsize = datavectors.shape[1]

s0max = datavectors[:,0].max() ; s0min = datavectors[:,0].min()
s1max = datavectors[:,1].max() ; s1min = datavectors[:,1].min()
s0 = s0max - s0min ; s1 = s1max - s1min

##########################
# Build the JTNN
##########################
jtp = JTNet.jtprob(vecsize, layers, penalty=penalty_const)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

##########################
# Log density function wrapper
##########################
sess = 0 # Will be set to tf.Session() later
mul = 1.0 # During optimization change this to -1.0
X = tf.placeholder(tf.float64, shape=(None, vecsize), name="X")
lpt = jtp.logprob(X)
def logprob(data_2d):
    # data_2d should have either shape (:,2) or (2)

    s2d = data_2d.shape
    if len(s2d) == 1:
        x = datavectors[val].reshape((1, vecsize)).copy()
        x[0,0:2] = data_2d[:]
    else:
        x = np.zeros((s2d[0],vecsize), dtype=np.float64)
        x[:,:] = datavectors[val].copy()
        x[:,0:2] = data_2d[:,:]
    
    x = scaler.transform(x)
    lp = sess.run(lpt, feed_dict={X: x})

    if len(s2d) == 1:
        lp = lp[0]
    
    return mul*lp

####################
# OK, let's go
####################
print(datavectors[val])
with tf.Session() as sess:
    saver.restore(sess, tf_chk_path)

    ## Find optimum predicted wind vector
    mul = -1.0
    v0 = datavectors[val, :2] # Initial guess is true value, cheating, I know, but harmless
    init_splx = np.zeros((3,2))
    init_splx[0] = v0
    init_splx[1] = v0 + np.array([s0*simplex_scale, 0])
    init_splx[2] = v0 + np.array([0, s1*simplex_scale])
    res = op.minimize(logprob, v0, method="Nelder-Mead", 
                       options={'initial_simplex':init_splx, 'disp': True})
    print(res)
    v_opt = res.x
    lp_opt = -res.fun  # this is the mode of the distribution?

    mul = 1.0
    ## Calculate grid of normalized probability densities
    # d0ctr = 0.5*(s0max+s0min) ; d1ctr = 0.5*(s1max+s1min)
    # d0hwid = 0.5*(s0max-s0min)*domain_scale ; d1hwid = 0.5*(s1max-s1min)*domain_scale
    # d0lo = d0ctr - d0hwid ; d0hi = d0ctr + d0hwid
    # d1lo = d1ctr - d1hwid ; d1hi = d1ctr + d1hwid
    # v0pts = np.linspace(d0lo, d0hi, nsamp, dtype=np.float64)
    # v1pts = np.linspace(d1lo, d1hi, nsamp, dtype=np.float64)
    v0pts = np.linspace(xdomain[0], xdomain[1], nsamp, dtype=np.float64)
    v1pts = np.linspace(ydomain[0], ydomain[1], nsamp, dtype=np.float64)
    cellarea = (v0pts[1]-v0pts[0]) * (v1pts[1] - v1pts[0])
    V0, V1 =np.meshgrid(v0pts, v1pts, indexing="ij")
    Vgrid = np.stack([V0,V1], axis=-1).reshape((-1,2))
    ldengrid = logprob(Vgrid).reshape((nsamp,nsamp))
    Pgrid = np.exp(ldengrid - lp_opt)
    Norm = Pgrid.sum() * cellarea
    Pgrid = Pgrid / Norm

print(datavectors[val])

## Calculate densities at requested contours
contours = np.array(contours)
contours.sort()
clevs = []
SP = Pgrid.flatten()
SP.sort()
SP = np.flip(SP)
cdf = 0.0
i=0
ic = 0
for c in contours:
    while cdf < c:
        cdf += SP[i]*cellarea
        i += 1
    clevs.append(SP[i])
clevs = np.flip(np.array(clevs))

## Calculate wind speed density
s_true = np.sqrt(datavectors[val,0]**2 + datavectors[val,1]**2)
dels = smax / s_samp
sarr = (np.arange(s_samp, dtype=np.float64) + 0.5) * dels
wspeed_2d = np.sqrt(V0**2+V1**2)
spden = np.zeros(s_samp)
for isamp in range(s_samp):
    vlo = sarr[isamp] - 0.5*dels
    vhi = sarr[isamp] + 0.5*dels
    spden[isamp] = Pgrid[np.logical_and(wspeed_2d > vlo, wspeed_2d <= vhi)].sum()
norm = spden.sum()*dels
spden = spden / norm

## Contour plot
plotname = cont_plot_stem + "%i.png" % val

fig = plt.figure()
fig.set_figwidth(10.0)
fig.set_figheight(10.0)
ax = fig.add_subplot(1,1,1)
plt.axes().set_aspect('equal', 'box')

p = ax.pcolormesh(V0, V1, ldengrid, cmap=cm.RdYlGn, vmin=ll_min)
fig.colorbar(p)

p = ax.contour(V0, V1, Pgrid, levels=clevs, colors="blue")
p.levels = np.flip(contours*100)
ax.clabel(p, fontsize=14, fmt="%3.1f%%")
true = ax.plot(datavectors[val,0], datavectors[val, 1], "r*", markersize=15.0, label = "3-Hr Delayed Observation")
mp = ax.plot(v_opt[0], v_opt[1], "b+", markersize=15.0, label="MAP Estimate")
ax.legend()
ax.set_xlabel(axis_labels[0],fontsize=14) ; ax.set_ylabel(axis_labels[1], fontsize=14)
ax.set_title(cplot_title, fontsize=14)

fig.savefig(plotname, format="png")
fig.clf()

# Windspeed probability distribution
plotname = speed_plot_stem + "%i.png" % val

ax = fig.add_subplot(1,1,1)
ax.plot(sarr, spden, "b-")
ax.set_xlabel("Wind Speed (m s$^{-1}$)", fontsize=14)
ax.set_ylabel("Probability Density", fontsize=14)
ax.set_title(splot_title, fontsize=14)
ax.axvline(s_true, color="green", linewidth=3, label="3-Hr Delayed Observation")
ax.legend(fontsize=14)
fig.savefig(plotname, format="png")

plt.close()
