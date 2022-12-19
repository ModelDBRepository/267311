# Copyright (c) 2022 Adrian Negrean
# negreanadrian@gmail.com
#
# Software released under MIT license, see license.txt for conditions
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as tck
import seaborn as sb

def distribute_subplots(max_nrows, max_ncols, nsubplots, layout = 'flexible'):
    """
    Distributes n subplots into multiple subplot grids of maximum size nrows x ncols.
    Distribution is done first by row, then by column with [0,0] corresponding to
    the upper left corner.

    Parameters
    ----------
    max_nrows, max_ncols : int
        Maximum number of rows and columns in a figure.

    nsubplots : int
        Total number of subplots to distribute.

    layout : str
        Choose 'fixed' to distribute plots on a fixed grid size on each page or
        choose 'flexible' to adjust the grid size depending on the number of plots
        to maximize space filling on each page.
    Returns
    -------
    list of tuple:
        Number of list elements is the number of figures needed to distribute plots. Each tuple is of the form
        ((nrows, ncols), [(plt_idx, row_1, col_1),...(plt_idx, row_N, col_N)])
        where:
        - first tuple element is the current plot index [0, nsubplots).
        - second tuple element is the grid size in # rows and # columns
        - third element is a list with distributed subplot 0-index coordinates
    """
    out = []
    nleft_to_assign = nsubplots
    curr_plt_idx = 0
    while nleft_to_assign:
        if layout == 'flexible':
            ncols = min(max_ncols, nleft_to_assign)
            nrows = min(max_nrows, int(math.ceil(nleft_to_assign/ncols)))
        elif layout == 'fixed':
            ncols = max_ncols
            nrows = max_nrows
        else:
            raise ValueError("Grid layout can be 'flexible' or 'fixed'.")

        # counter for number of subplots assigned for current figure
        plt_ctr = 0
        fig_plots = []
        nplts = min(nleft_to_assign, ncols*nrows)
        while plt_ctr < nplts:
            col_idx = plt_ctr%ncols
            row_idx = int(plt_ctr/ncols)
            fig_plots.append((curr_plt_idx,row_idx,col_idx))
            plt_ctr += 1
            curr_plt_idx += 1

        out.append(((nrows,ncols),fig_plots))
        nleft_to_assign -= nplts

    return out

def rec_grid_plot(pset, mruns, recpar, dt, secseg_names_filter = []):
    """
    Plot recorded variables over a grid.

    Parameters
    ----------
    pset : dict
        Plot settings.
    mruns : numpy nd.array of dict
        Recorded parameters.
    recpar : str
        Name of recorded parameter to plot.
    dt : float
        Time step in [ms].
    secseg_names_filter : list of str
        Filter sections and segment names.
    Returns
    -------
    fig : matplotlib.Figure
    """
    secseg_names = mruns.flat[0]["rec"][recpar].keys()

    # filter sections and segments to plot from all recorded sections and segments
    if secseg_names_filter:
        secseg_names = [x for x in _expand_secseg_list(secsegs = secseg_names_filter, env_set = env_set) if x in secseg_names]

    if len(mruns.shape) == 1:
        # number of columns in the grid plot
        ncols = 1
        # number of rows in the grid plot
        nrows = len(secseg_names)
        # adjust dimensions to make it compatible with plotting
        mruns = np.expand_dims(mruns,0)
    elif len(mruns.shape) == 2:
        # number of columns in the grid plot
        # organize columns by the first axis of the parameter sweep (last axis will be color hue in the multiline plot)
        ncols = mruns.shape[0]
        # number of rows in the grid plot
        # plot on each row parameters recorded in a certain section or segment
        nrows = len(secseg_names)
    else:
        raise Exception("Parameter sweep dimension {} is not compatible with grid plot.".format(len(mruns.shape)))

    # number of colors in the palette is determined by the last axis dimension of mruns, i.e. the last axis of the parameter sweep
    cmap = sb.cubehelix_palette(n_colors = mruns.shape[-1], start = 2.7, rot = 0, dark = 0.4, light = .9, reverse = False)

    fig, ax = plt.subplots(nrows, ncols, squeeze = False, sharex = True, sharey = True)
    for row_idx, sec_seg_name in enumerate(secseg_names):
        for col_idx in range(ncols):
            for color_idx, color in enumerate(cmap):
                y = mruns[col_idx,color_idx]["rec"][recpar][sec_seg_name][0]
                x = dt*np.arange(len(y))

                if color_idx < len(cmap)-1:
                    ax[row_idx,col_idx].plot(x,y, color = cmap[color_idx])
                else:
                    ax[row_idx,col_idx].plot(x,y, color = cmap[color_idx], label = recpar)
            if not row_idx and not col_idx:
                ax[row_idx,col_idx].legend()
            # set range of axes
            if "xlim" in pset["display"]:
                ax[row_idx,col_idx].set_xlim(pset["display"]["xlim"])
            if "ylim" in pset["display"]:
                ax[row_idx,col_idx].set_ylim(pset["display"]["ylim"])
            # add y-axis minor ticks
            ax[row_idx,col_idx].yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax[row_idx, 0].set_ylabel(sec_seg_name)

    return fig

def plot_dendrogram(dtree, secdata, ax, linestyle = "-", color = (0,0,1,1), alpha = None):
    """
    Plots segment level parameter as a function of distance using a dendrogram style display

    Parameters
    ----------
    dtree : dict
        Dendrogram structure.
    secdata : dict
        Section data to plot. Keys are section names and values are 1D numpy arrays of length
        equal to the number of segments within a section.
    """
    def _rec_dend_plot(node, secdata, ax, linestyle, color, dist = 0, alpha = None):
        """
        Recursively plot dendrogram.
        """
        # iterate over parent sections
        for pkey in node:
            # distance to end of parent section
            dist_to_parent_1end = dist+node[pkey][0][-1]
            # plot parent section input impedance
            ax.plot(dist+node[pkey][0], secdata[pkey][0,:,0], linestyle = linestyle, color = color, alpha = alpha)
            
            # connect end of parent section to start of child section
            for ckey in node[pkey][1]:
                dist_to_child_0end = dist_to_parent_1end+node[pkey][1][ckey][0][0]    
                ax.plot([dist_to_parent_1end, dist_to_child_0end], [secdata[pkey][0,-1,0], secdata[ckey][0,0,0] ], linestyle = linestyle,
                    color = color, alpha = alpha)
            _rec_dend_plot(node = node[pkey][1], secdata = secdata, ax = ax, linestyle = linestyle, color = color, dist = dist_to_parent_1end,
                alpha = alpha)

            

    _rec_dend_plot(node = dtree, secdata = secdata, ax = ax, linestyle = linestyle, color = color, alpha = alpha)

    
    