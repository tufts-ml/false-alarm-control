import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import expit as logistic_sigmoid

from BCE import make_loss_and_grad_for_dataset
from binary_clf_utils import calc_binary_clf_perf, calc_decision_score

def pretty_plot_decision_boundary(
        w_D, x_ND, y_N,
        title_str='',
        contour_values_to_plot=[0.0],
        G=51, H=51, figsize=(3,3),
        transparency_level=0.4,
        x1_lims=(-3, 3), x2_lims=(-3, 3),
        x1_ticks=[-2, 0, 2], x2_ticks=[-2, 0, 2],
        decision_thresh=0.0, ax=None): 
    bmask_pos_N = y_N==1
    x_pos_ND = x_ND[bmask_pos_N]
    y_pos_N = y_N[bmask_pos_N]

    bmask_neg_N = y_N==0
    x_neg_ND = x_ND[bmask_neg_N]
    y_neg_N = y_N[bmask_neg_N]
    bce_loss, _ = make_loss_and_grad_for_dataset(x_ND, y_N)

    redblue_colors = plt.cm.RdBu(np.linspace(0, 1, 101))
    redblue_cmap_r = matplotlib.colors.ListedColormap(redblue_colors[::-1])
    
    # Create empty figure with slot for colorbar
    if ax is None:
        fig, ax_h = plt.subplots(figsize=figsize)
    else:
        ax_h = ax
    divider = make_axes_locatable(ax_h)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    ax_h.set_xlim(x1_lims)
    ax_h.set_ylim(x2_lims)

    # Plot the raw data, colored by true label
    ax_h.plot(x_neg_ND[:,0], x_neg_ND[:,1], 'x', color=redblue_colors[-1])
    ax_h.plot(x_pos_ND[:,0], x_pos_ND[:,1], '+', color=redblue_colors[0])
    ax_h.set_xlabel(r'$x_1$')
    ax_h.set_ylabel(r'$x_2$')
    ax_h.set_xticks(x1_ticks)
    ax_h.set_yticks(x2_ticks)
    
    # Create grid of input features covering the entire feature space
    # J = G * H
    x0_G = np.linspace(*ax_h.get_xlim(), num=G)
    x1_H = np.linspace(*ax_h.get_ylim(), num=H)
    G = x0_G.size
    H = x1_H.size
    X0_HG, X1_HG = np.meshgrid(x0_G, x1_H)
    assert X0_HG.shape == (H, G)
    x_JD = np.hstack([X0_HG.flatten()[:,np.newaxis], X1_HG.flatten()[:,np.newaxis]])
    assert np.allclose(X0_HG, x_JD[:,0].reshape((H,G)))    
    
    # Compute the decision score at all grid cells
    w_3 = np.hstack([w_D[:2], w_D[-1]])
    z_J = calc_decision_score(x_JD, w_3)
    Z_HG = z_J.reshape((H,G))
    
    # Draw the level contours for the decision boundary
    # Want grays that are not too light and not too dark
    L = np.maximum(len(contour_values_to_plot), 11)
    level_colors = plt.cm.Greys(np.linspace(0, 1, L))
    m = L // 2
    nrem = len(contour_values_to_plot)
    mlow = m - nrem // 2
    mhigh = m + nrem // 2 + 1
    if mhigh - mlow < len(contour_values_to_plot):
        mhigh += 1
    levels_gray_cmap = matplotlib.colors.ListedColormap(level_colors[mlow:mhigh])
    ax_h.contour(
        X0_HG, X1_HG, Z_HG,
        levels=contour_values_to_plot,
        cmap=levels_gray_cmap,
        vmin=-2, vmax=+2);

    # Draw predicted probabilities as a colored grid
    # Remember, (left, right, bottom, top)  is how to interpret 'extent'
    left, right = x0_G[0], x0_G[-1]
    bottom, top = x1_H[0], x1_H[-1]
    im = ax_h.imshow(
        logistic_sigmoid(Z_HG),
        alpha=transparency_level, cmap=redblue_cmap_r,
        interpolation='nearest',
        origin='lower', # this is crucial 
        extent=(left, right, bottom, top),
        vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.draw_all()

    # Evaluate performance at a specific threshold
    z_N = calc_decision_score(x_ND, w_D)
    yhat_N = np.float64(z_N >= decision_thresh)
    perfdict = calc_binary_clf_perf(y_N, yhat_N)
#     ax_h.set_title("%s\n Prec:%.2f  Rec:%.2f  BCE:% .1f\n" % (
#         title_str,
#         perfdict['precision'],
#         perfdict['recall'], bce_loss(w_D)))
    
    ax_h.set_title("%s\n Prec:%.2f  Rec:%.2f  Runtime:3000 sec\n" % (
    title_str,
    perfdict['precision'],
    perfdict['recall']))

    return perfdict, im, cbar

def calc_precision_recall(w_D, x_ND, y_N, decision_thresh=0.0): 
    # Evaluate performance at a specific threshold
    z_N = calc_decision_score(x_ND, w_D)
    yhat_N = np.float64(z_N >= decision_thresh)
    perfdict = calc_binary_clf_perf(y_N, yhat_N)
    return perfdict
