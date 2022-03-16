import matplotlib.pyplot as plt

fsize=16
fsize_l=12
lw_ax=2
lw_plot=1
msize=18
fig = plt.figure()
fig.set_figwidth(18.0)
fig.set_figheight(12.0)
import matplotlib 

def tracker_plots(ocdict, plotpath, plotpenalty=True):
# ocdict keys become variables:
    globals().update(ocdict)
    cols = 2 if plotpenalty else 1

# Loss plot panel
    ax = fig.add_subplot(2,cols,1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw_ax)

    ax.plot(epoch_oc, trloss_oc, "b+-", linewidth=lw_plot, label="Training Loss")
    ax.plot(epoch_oc, vloss_oc, "gx--", linewidth=lw_plot, label="Val Loss")
    #ax.plot(epoch_oc, floss_oc, "r.:", linewidth=lw_plot, label="F-Test Loss")
    #ax.plot(epoch_oc, tloss_oc, "c1-", linewidth=lw_plot, label="Test Loss")
    ax.set_xlabel("Training Epoch", size=fsize)
    ax.set_ylabel("Loss", size=fsize)
    ax.set_yscale('log')
    ax.legend()

# Log-Jacobian plot panel
    ax = fig.add_subplot(2,cols,2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw_ax)

    ax.plot(epoch_oc, trljac_oc, "b+-", linewidth=lw_plot, label="Training Log-Jacobian")
    ax.plot(epoch_oc, vljac_oc, "gx--", linewidth=lw_plot, label="Val Log-Jacobian")
    #ax.plot(epoch_oc, fljac_oc, "r.:", linewidth=lw_plot, label="F-Test Log-Jacobian")
    #ax.plot(epoch_oc, tljac_oc, "c1-", linewidth=lw_plot, label="Test Log-Jacobian")
    ax.set_xlabel("Training Epoch", size=fsize)
    ax.set_ylabel("Log Jacobian", size=fsize)
    ax.legend()

# Penalty plot panel
    if plotpenalty:
        ax = fig.add_subplot(2,cols,5)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(lw_ax)

        ax.plot(epoch, penalty, "b+-", linewidth=lw_plot, label="Training Penalty")
        ax.plot(epoch, bpenalty, "gx--", linewidth=lw_plot, label="B-Test Penalty")
        ax.plot(epoch, fpenalty, "r.:", linewidth=lw_plot, label="F-Test Penalty")
        ax.plot(epoch, vpenalty, "c1-", linewidth=lw_plot, label="V-Test Penalty")
        ax.set_xlabel("Training Epoch", size=fsize)
        ax.set_ylabel("Penalty", size=fsize)
        ax.legend()


    plt.savefig(plotpath,format="png")

    plt.clf()
