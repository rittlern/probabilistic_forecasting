#Module for creating staircase plots
#
#Andrew Pensoneault
import matplotlib


font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 6}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

def staircase_plot(inverse_data, figname, hist_scalex, hist_scaley, bin_num, xtrue = [1,1,-2,-1,1]):
    ndim = inverse_data.shape[1]
    fig = plt.figure()
    fig.set_figwidth(8.0)
    fig.set_figheight(8.0)
    for k in range(ndim):
        print(k)
        ax = fig.add_subplot(ndim,ndim,1+k+k*ndim)
        ax.hist(inverse_data[:,k], density="True", facecolor='green', alpha=0.75, bins=bin_num)
        plt.axvline(x=xtrue[k],color='red',linewidth=2)
        ax.yaxis.tick_right()
        ax.set_xlim(hist_scalex[k])
        ax.set_ylim(hist_scaley[k])
        plt.tick_params(
                        axis='x',       # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
        if k == ndim-1:
            plt.tick_params(
                            axis='x',       # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=True,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=True) # labels along the bottom edge are off

        for kk in range(k):
            print(kk)
            ax = fig.add_subplot(ndim,ndim,k*ndim+kk+1)
            bindat, xedge, yedge = np.histogram2d(inverse_data[:,kk],inverse_data[:,k],bins = 100,range = [hist_scalex[kk],hist_scalex[k]],density=True)
                

            sortden = np.flip(np.sort(np.ndarray.flatten(bindat)))
            cumsumden = np.cumsum(sortden)
            cumsumden = cumsumden/cumsumden[-1]
            d68 = sortden[np.abs(cumsumden - .68).argmin()]
            d95 = sortden[np.abs(cumsumden - .95).argmin()]
            d99 = sortden[np.abs(cumsumden - .997).argmin()]
            levels =  [d99,d95,d68] 
            ax.contour((xedge[0:-1]), (yedge[0:-1]), bindat.T, levels = levels,colors = ['k','r','b'],linewidths=(1,), origin="lower")

            ax.plot(xtrue[kk],xtrue[k],'r*',markersize = 8)
            ax.set_xlim(hist_scalex[kk])
            ax.set_ylim(hist_scalex[k])
            plt.tick_params(
                            axis='both',       # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False,
                            left=False,      # ticks along the left edge are off
                            right=False,         # ticks along the right edge are off
                            labelleft=False) # labels along the left edge are off
            if kk == 0:
                ax.set_ylabel("$x_{%d}$" % k,fontsize=18)
                plt.tick_params(
                                axis='y',       # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                left=True,      # ticks along the bottom edge are off
                                right=False,         # ticks along the  edge are off
                                labelleft=True) # labels along the left edge are off
                ax.get_yaxis().set_label_coords(-0.3,0.5)
            if k == ndim-1:
                ax.set_xlabel("$x_{%d}$" % kk,fontsize=18)
                plt.tick_params(
                                axis='x',       # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=True,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=True) # labels along the bottom edge are off
                
    fig.savefig(figname, format="png")
    fig.clf()
    plt.close()


def staircase_plot_simple(inverse_data, figname, ndim, scales, bin_num, response_names=None, pred_names=None):
    fig = plt.figure()
    fig.set_figwidth(10.0)
    fig.set_figheight(10.0)

    if response_names is not None and pred_names is not None:
        assert (len(response_names) + len(pred_names) == ndim)

    for k in range(ndim):
        #print(k)
        ax = fig.add_subplot(ndim, ndim, 1 + k + k * ndim)
        ax.hist(inverse_data[:, k], density="True", facecolor='green', alpha=0.75, bins=bin_num)
        ax.yaxis.tick_right()
        if scales is not None:
            ax.set_xlim(scales[k])
        if k < ndim - 1:
            ax.tick_params(bottom=False, labelbottom=False)

        for kk in range(k):
            #print(kk)
            ax = fig.add_subplot(ndim, ndim, k * ndim + kk + 1)
            KDE = gaussian_kde(inverse_data[:, (kk, k)].T)
            z = KDE(inverse_data[:, (kk, k)].T)
            ax.scatter(inverse_data[:, kk], inverse_data[:, k], c=z, s=10, edgecolor='')
            if scales is not None:
                ax.set_xlim(scales[kk])
                ax.set_ylim(scales[k])
            if kk == 0:
                if response_names is not None and pred_names is not None:
                    if k < len(response_names):
                        ax.set_ylabel(response_names[k], fontsize=18)
                    else:
                        ax.set_ylabel(pred_names[k - len(response_names)], fontsize=18)
                else:
                    ax.set_ylabel("$x_{%d}$" % k, fontsize=18)
            else:
                ax.tick_params(left=False, labelleft=False)

            if k == ndim - 1:
                if response_names is not None and pred_names is not None:
                    if kk < len(response_names):
                        ax.set_xlabel(response_names[kk], fontsize=18)
                    else:
                        ax.set_xlabel(pred_names[kk - len(response_names)], fontsize=18)
                else:
                    ax.set_xlabel("$x_{%d}$" % kk, fontsize=18)
            else:
                ax.tick_params(bottom=False, labelbottom=False)

    fig.savefig(figname, format="png")
    fig.clf()
    plt.close()
