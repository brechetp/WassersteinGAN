__doc__ = '''Module to help visualization and plotting'''

import torch
import matplotlib
matplotlib.use('Qt5Agg')  # for remote working
import matplotlib.pyplot as plt
import utils
import pdb
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D

def hist(data, **kwargs):
    '''Histogram of the data'''
    _,ax = new_figure(**kwargs)
    kw_hist = utils.filter_kwargs(kwargs, ax.hist, plt.hist)
    ax.hist(data, **kw_hist)
    ax.legend()
    add_text(ax, **kwargs)
    return

def hist_and_save(path, data, **kwargs):
    '''Saves the histogram'''
    hist(data, **kwargs)
    save(path)
    return

def quiver(X, Y, U, V, **kwargs):
    '''Quiver plot for the vector in origin+dx,dy'''
    _, ax =new_figure(**kwargs)
    C = kwargs.pop('C', None)  # for the color of the plot
    kw_quiver = utils.filter_kwargs(kwargs, ax.quiver)
    # ax.quiver(X, Y, U, V, C, **kw_quiver)
    ax.quiver(X, Y, U, V, scale=1, **kw_quiver)
    add_text(ax, **kwargs)
    return

def quiver_and_save(path, **kwargs):
    '''quiver plot and save'''
    quiver(**kwargs)
    save(path)
    return

def plot(t, x, **kwargs):
    '''Plot x wrt t

    :kwargs: arguments for the pyplot function
    '''
    ax = kwargs.pop('ax', None)# the default is having a new figure
    if ax is None:
        _, ax = new_figure(**kwargs)
    kwargs_plot = utils.filter_kwargs(kwargs, ax.plot)
    # pdb.set_trace()
    ax.plot(t, x, **kwargs_plot)
    add_text(ax, **kwargs)
    return

def new_figure(**kwargs):
    '''Confgures a new figure, should be called before any plotting'''
    # title = kwargs.pop('title', '')
    subplots = kwargs.pop('subplots', (1, 1))  # should be iterable
    # projection = kwargs.pop('projection', None)
    nrows, ncols= subplots
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    fig = plt.figure()
    subplot_kw = utils.filter_kwargs(kwargs, fig.add_subplot, add_kwargs=['projection'])
    plt.close('all')
    # if projection is None:
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=subplot_kw)
    # else:
        # fig, axes = plt.subplots(nrows, ncols, projection=projection, subplot_kw=subplot_kw)
    # axes.set_title(title)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    return fig, axes

def add_text(ax, **kw_ax):
    '''Adds a text in the artist object'''
    text = kw_ax.pop('text', None)
    if text is not None:
        anchored_text = AnchoredText(text, loc='upper right', borderpad=-1.5)
        ax.add_artist(anchored_text)
    return ax

def save(path, close=True):
    '''Save the figure to the path and cleans the figures'''
    plt.savefig(path)
    if close:
        plt.clf()
    return

def plot_and_save(path, t, x, **kwargs):
    '''Plot and save the plot'''
    plot(t, x, **kwargs)
    save(path)
    return

def scatter(x, y, ax, **kwargs):
    '''Scatter plot'''
    # pdb.set_trace()
    kwargs_scatter = utils.filter_kwargs(kwargs, plt.scatter, ax.scatter, add_kwargs=['zorder', 'color'])
    ax.scatter(x, y, **kwargs_scatter)
    ax = add_text(ax, **kwargs)
    ax.legend()
    return ax

def imshow(mat, **kwargs):
    '''Show the matrix on the graph'''
    fig, ax = new_figure(**kwargs)
    kw_imshow = utils.filter_kwargs(kwargs, ax.imshow, plt.imshow)
    im = ax.imshow(mat, **kw_imshow)
    ax = add_text(ax, **kwargs)
    fig.colorbar(im)
    return im

def imshow_and_save(path, mat, **kwargs):
    imshow(mat, **kwargs)
    save(path)
    return

def scatter_and_save(path, x, y, **kwargs):
    '''Scatter and save'''
    _, ax = new_figure(**kwargs)
    scatter(x, y, ax, **kwargs)
    save(path)
    return

def scatter_on_same_ax(x, y, ax=None, **kwargs):
    '''Scatter on the same figure'''
    if ax is None:  # on the first call we don't have a ax yet
        _, ax = new_figure(**kwargs)
    ax = scatter(x, y, ax=ax, **kwargs)
    return ax

def scatter_3d(x, y, z, **kwargs):
    '''3d plot'''
    _,ax = new_figure(projection='3d', **kwargs)
    kwargs_scatter = utils.filter_kwargs(kwargs, ax.scatter, add_kwargs=['label'])  # we filter the keyword arguments
    ax.scatter(x, y, z, **kwargs_scatter)
    ax = add_text(ax, **kwargs)
    ax.set_xlabel(kwargs.pop('xlabel', 'x'))
    ax.set_ylabel(kwargs.pop('ylabel', 'y'))
    ax.set_zlabel(kwargs.pop('zlabel', 'z'))
    return ax

def scatter_3d_and_save(path, x, y, z, **kwargs):
    scatter_3d(x, y, z, **kwargs)
    save(path)
    return
