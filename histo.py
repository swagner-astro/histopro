"""
Useful tools for plotting histograms
each function plots on individual axis, see:
https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1
and general info at:
https://towardsdatascience.com/plot-organization-in-matplotlib-your-one-stop-guide-if-you-are-reading-this-it-is-probably-f79c2dcbc801

1. various binnigs
2. logarithmic histogram
3. multi histograms
4. fit histogram
5. note for multible custom plots (many **kwargses)

"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.visualization.hist as fancy_hist     # https://docs.astropy.org/en/stable/api/astropy.visualization.hist.html
from lmfit import Model #https://lmfit.github.io/lmfit-py/
from numpy import exp, loadtxt, pi, sqrt


#---------------------------------------------------------------------------------------------
"""
convenient way for different binnings and histo representations
tbd: using fancy_hist is probably inefficient as it makes unnecessary plot
"""
def bay_bin(data, ax=None, **histo_kwargs):
    values, bay_bins, p = fancy_hist(data,bins='blocks')
    plt.close() # maybe problems if many plots
    if ax is None:
        ax = plt.gca()
    values, bay_bins, p = ax.hist(data, bins=bay_bins, **histo_kwargs)
    return [values, bay_bins, p]


def knuth_bin(data, ax=None, **histo_kwargs):
    values, knuth_bins, p = fancy_hist(data, bins='knuth')
    plt.close() # maybe problems if many plots
    if ax is None:
        ax = plt.gca()
    values, knuth_bins, p = ax.hist(data, bins=knuth_bins, **histo_kwargs)
    return [values, knuth_bins, p]

def rainbowew(data, bins, package_size=1, ax=None):
    """
    plot color coded hist from timeseries
    tbd: plot this on individual axis
    """
    if ax is None:
        ax = plt.gca()
    L = int(len(data) / package_size)
    color = plt.cm.rainbow(np.linspace(0,1,L))
    for t,c in zip(range(L), color):
        histo = ax.hist(data[0 : t*package_size], bins=bins, zorder=L-t, color=c)
    return ax

def rainbowew_timeseries(data, data_error=None, time=None, package_size=1,
                         ax=None, **plot_kwargs):
    """
    plot color coded time series
    """
    if time is None:
        time = np.arange(0, len(data), 1)
    if ax is None:
        ax = plt.gca()
    L = int(len(data)/package_size)
    color = plt.cm.rainbow(np.linspace(0,1,L))
    if data_error:
        for t,c in zip(range(L),color):
            ax.errorbar(x=time[0:t*package_size], y=data[0:t*package_size], 
                        yerr=data_error[0:t*package_size],
                        ecolor=c, zorder=L-t, color=c, #**plot_kwargs) 
                        elinewidth=0.1, linewidth=0, marker='+', markersize=1, **plot_kwargs)
        ax.errorbar(x=time, y=data, yerr=data_error, ecolor=c, zorder=0, color=c, #**plot_kwargs)
                    elinewidth=0.1, linewidth=0, marker='+', markersize=1, **plot_kwargs)
        return ax
    else:
        for t,c in zip(range(L),color):
            ax.plot(time[0:t*package_size], data[0:t*package_size],
                    zorder=L-t, color=c, #**plot_kwargs) 
                    linewidth=0, marker='+', markersize=1, **plot_kwargs)
        ax.plot(time, data, zorder=0, color=c, #**plot_kwargs)
                linewidth=0, marker='+', markersize=1, **plot_kwargs)
        return ax


#---------------------------------------------------------------------------------------------
"""
logarithmic histograms
Note: use kwarg log=True or plt.yscale('log') to plot y-axis in logscale
Note: plt.xscale('log') produces bins with constant (linear) width on logscale axis
"""
def make_logbins(data, N_bins):
    bin_start = np.min(data)
    bin_end = np.max(data)
    logbins = np.logspace(np.log10(bin_start), np.log10(bin_end), N_bins)
    return logbins

def log(data, N_bins=50, ax=None, **histo_kwargs):
    # x-axis = logarithmic scale
    logbins = make_logbins(data, N_bins)
    if ax is None:
        ax = plt.gca()
    values, bins, p = ax.hist(data, logbins, **histo_kwargs)
    ax.set_xscale('log')
    return [values, bins, p]

def log_but_lin(data, N_bins=50, ax=None, **histo_kwargs):
    # x-axis = linear scale of logarithmic values
    logbins = make_logbins(data, N_bins)
    if ax is None:
        ax = plt.gca()
    values, bins, p = ax.hist(np.log10(data), bins=np.log10(logbins), **histo_kwargs)
    return [values, bins, p]


#---------------------------------------------------------------------------------------------
"""
multi histograms of several data arrays
eg KA94 -> show time evolution of distribution
"""
def make_multi_bins(data_sets, bin_width):
    bin_start = np.min([np.min(data_sets[i]) for i in range(len(data_sets))])
    bin_end = np.max([np.max(data_sets[i]) for i in range(len(data_sets))])
    bins = np.arange(bin_start, bin_end, bin_width)
    return bins

def multi(data_sets, bin_width, labels=None, ax=None, **hist_kwargs):
    """
    plot histogram of each array with the same bins
    data_sets = list of data arrays
    labels = label of each array
    """
    bins = make_multi_bins(data_sets, bin_width)
    if labels is None:
        labels = [i for i in range(len(data_sets))]
    if ax is None:
        ax = plt.gca()
    for i,data in enumerate(data_sets):
        ax.hist(data, bins, label=labels[i], histtype='step', **hist_kwargs)
    return (bins, ax) #tbd: return histo values in array

def make_multi_logbins(data_sets, N_bins):
    bin_start = np.min([np.min(data_sets[i]) for i in range(len(data_sets))])
    bin_end = np.max([np.max(data_sets[i]) for i in range(len(data_sets))])
    logbins = np.logspace(np.log10(bin_start), np.log10(bin_end), N_bins)
    return logbins

def multi_log(data_sets, labels=None, N_bins=50, ax=None, **hist_kwargs):
    # logarithmic scale
    logbins = make_multi_logbins(data_sets, N_bins)
    if labels is None:
        labels = [i for i in range(len(data_sets))]
    if ax is None:
        ax = plt.gca()
    for i,data in enumerate(data_sets):
        ax.hist(data, logbins, label=labels[i], histtype='step', **hist_kwargs)
    ax.set_xscale('log')
    return ax #tbd: return histo values in array

def multi_log_but_lin(data_sets, N_bins=50, labels=None, ax=None, **hist_kwargs):
    # linear scale of logarithmic values 
    logbins = make_multi_logbins(data_sets, N_bins)
    if labels is None:
        labels = [i for i in range(len(data_sets))]
    if ax is None:
        ax = plt.gca()
    for i,data in enumerate(data_sets):
        ax.hist(np.log10(data), bins=np.log10(logbins), label=labels[i], 
                histtype='step', **hist_kwargs)
    return ax #tbd: return histo values in array


#---------------------------------------------------------------------------------------------
"""
fit data of histogram with Gaussian
error on bins (weight) -> use Poisson error: sigma_bin = sqrt(N_inbin)
"""
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

def make_gauss_fit(values, bins):
    left = bins[:-1]
    right = bins[1:]
    x = (right-left)/2 + left 
    y = values
    gmodel = Model(gaussian)
    params = gmodel.make_params(cen=np.mean(x), 
                                amp=np.max(y), wid=np.std(x))#1)
    result = gmodel.fit(y,params,x=x, weights=1/np.where(y == 0, 1, y)) #Poisson error
    amp = result.params['amp'].value
    mu = result.params['cen'].value  
    sigma = result.params['wid'].value
    return(amp, mu, abs(sigma), result.chisqr, result.redchi)

def gauss_fit(values, bins, plotpoints=200, ax=None, **histo_kwargs):
    amp, mu, sigma, chisqr, redchi = make_gauss_fit(values, bins)
    x_plot = np.linspace(bins[0], bins[-1], plotpoints)
    y_plot = gaussian(x_plot, amp, mu, sigma)
    if ax is None:
        ax = plt.gca()
    ax.plot(x_plot,y_plot, marker='', color='k', zorder=13148, **histo_kwargs)
    return(amp, mu, abs(sigma), chisqr, redchi)


#---------------------------------------------------------------------------------------------
"""
when plotting many custom plots, i.e. several **kwargs use this
"""
"""
def multiple_custom_plots(x, y, ax=None, plt_kwargs={}, sct_kwargs={}):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **plt_kwargs)
    ax.scatter(x, y, **sct_kwargs)
    return(ax)
plot_params = {'linewidth': 2, 'c': 'g', 'linestyle':'--'}
scatter_params = {'c':'red', 'marker':'+', 's':100}
xdata = [1, 2]
ydata = [10, 20]
plt.figure(figsize=(10, 5))
multiple_custom_plots(xdata, ydata, plt_kwargs=plot_params, sct_kwargs=scatter_params)
plt.show()
"""
