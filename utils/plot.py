import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Sequence, Tuple, Union
import swyft as sl
import numpy as np
from tabulate import tabulate
from scipy import stats
from sys import float_info
from matplotlib.ticker import FormatStrFormatter

tabData = np.zeros((6, 7))

def corner(
    logratios,
    parnames,
    bins=100,
    truth=None,
    figsize=(7, 7),
    color="k",
    labels=None,
    label_args={},
    contours_1d: bool = True,
    fig=None,
    labeler=None,
    smooth=0.0,
    cmap="gray_r",
    cmap_2="k",
    vtrue=None
) -> None:
    """Make a beautiful corner plot.
    Args:
        samples: Samples from `swyft.Posteriors.sample`
        pois: List of parameters of interest
        truth: Ground truth vector
        bins: Number of bins used for histograms.
        figsize: Size of figure
        color: Color
        labels: Custom labels (default is parameter names)
        label_args: Custom label arguments
        contours_1d: Plot 1-dim contours
        fig: Figure instance
    """
    K = len(parnames)
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=figsize)
    else:
        axes = np.array(fig.get_axes()).reshape((K, K))
    lb = 0.125
    tr = 0.9
    whspace = 0.1
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    diagnostics = {}

    if labeler is not None:
        labels = [labeler.get(k, k) for k in parnames]
    else:
        labels = labels
        
    for i in range(len(labels)):
        axes[i, 0].get_yaxis().set_label_coords(-0.38, 0.5)
        axes[i, 0].set_xticks([30, 60, 90])
        axes[i, 2].set_xticks([10, 15, 20])
        axes[i, 4].set_xticks([0.2, 0.5, 0.8])
        axes[i, 5].set_xticks([0, 1, 2])
        axes[2, i].set_yticks([10, 15, 20])
        axes[4, i].set_yticks([0.2, 0.5, 0.8])
        axes[5, i].set_yticks([0, 1, 2])
        
    axes[0, 0].set_ylabel("")
    axes[0, 0].set_yticks([])
    
    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=10, length=4)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='minor', labelsize=8, length=2)
            # Switch off upper left triangle
            if i < j:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                continue

            # Formatting labels
            if j > 0 or i == 0:
                ax.set_yticklabels([])
                # ax.set_yticks([])
            if i < K - 1:
                ax.set_xticklabels([])
                # ax.set_xticks([])
            if i == K - 1:
                ax.set_xlabel(labels[j], **label_args)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], **label_args)

            # 2-dim plots
            if j < i:
                try:
                    ax.plot((vtrue[j]), (vtrue[i]), "s", color="k", markersize=1.3)
                    ax.axvline(vtrue[j], c='k', lw=0.7, ls="--")
                    ax.axhline(vtrue[i], c='k', lw=0.7, ls="--")
                    ret = plot_2d(
                        logratios,
                        parnames[j],
                        parnames[i],
                        ax=ax,
                        color=color,
                        bins=bins,
                        smooth=smooth,
                        cmap=cmap
                    )
                except sl.SwyftParameterError:
                    pass
            if j == i:
                try:
                    ax.axvline(vtrue[j], c='k', lw=0.7, ls="--")
                    ax.set_yticks([])
                    m, m1s, p1s, m2s, p2s, m3s, p3s = plot_1d(
                        logratios,
                        parnames[i],
                        ax=ax,
                        color=color,
                        bins=bins,
                        contours=contours_1d,
                        smooth=smooth,
                        cmap=cmap_2
                    )
                    for index, info in zip(range(7), [m, m1s, p1s, m2s, p2s, m3s, p3s]):
                        tabData[i, index] = info
                except sl.SwyftParameterError:
                    pass
    print(tabulate(tabData, headers=["mean", "-1s", "+1s", "-2s", "+2s", "-3s", "+3s"], tablefmt="mixed_grid"))
    return fig


def get_HDI_thresholds(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = x.flatten()
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def plot_2d(
    logratios,
    parname1,
    parname2,
    ax=plt,
    bins=100,
    color="k",
    cmap="gray_r",
    smooth=0.0,
):
    """Plot 2-dimensional posteriors."""
    counts, xy = sl.lightning.utils.get_pdf(
        logratios, [parname1, parname2], bins=bins, smooth=smooth
    )
    xbins = xy[:, 0]
    ybins = xy[:, 1]

    levels = sorted(get_HDI_thresholds(counts))
    ax.contourf(
        counts.T,
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
        levels=levels,
        # linestyles=[":", "--", "-"],
        # colors=color,
        cmap=cmap,
        extend="max"
    )
    ax.set_xlim([xbins.min(), xbins.max()])
    ax.set_ylim([ybins.min(), ybins.max()])


def plot_1d(
    logratios,
    parname,
    weights_key=None,
    ax=plt,
    grid_interpolate=False,
    bins=100,
    color="k",
    contours=True,
    smooth=0.0,
    cmap=None
):
    """Plot 1-dimensional posteriors."""
    v, zm = sl.lightning.utils.get_pdf(logratios, parname, bins=bins, smooth=smooth)
    zm = zm[:, 0]
    levels = sorted(get_HDI_thresholds(v))
    if contours:
        m, m1s, p1s, m2s, p2s, m3s, p3s = contour1d(zm, v, levels, ax=ax, cmap=cmap)
    ax.plot(zm, v, color=color, lw=1)
    ax.set_xlim([zm.min(), zm.max()])
    ax.set_ylim([-v.max() * 0.05, v.max() * 1.1])
    return m, m1s, p1s, m2s, p2s, m3s, p3s


def contour1d(z, v, levels, ax=plt, cmap=None, linestyles=None, color=None, **kwargs):
    y0 = -1.0 * v.max()
    y1 = 5.0 * v.max()
    mode = z[np.where(v==v.max())].item()
    
    all_levels_1s = z[np.where(v>levels[2])]
    all_levels_2s = z[np.where(v>levels[1])]
    all_levels_3s = z[np.where(v>levels[0])]
    
    min_1s = all_levels_1s[0].item()
    max_1s = all_levels_1s[-1].item()
    
    min_2s = all_levels_2s[0].item()
    max_2s = all_levels_2s[-1].item()
    
    min_3s = all_levels_3s[0].item()
    max_3s = all_levels_3s[-1].item()
    
    ax.fill_between(z, y0, y1, where=v > levels[0], color=cmap[1], alpha=0.3)
    ax.fill_between(z, y0, y1, where=v > levels[1], color=cmap[2], alpha=0.4)
    ax.fill_between(z, y0, y1, where=v > levels[2], color=cmap[4], alpha=0.5)
    return mode, mode-min_1s, max_1s-mode, mode-min_2s, max_2s-mode, mode-min_3s, max_3s-mode

def get_alpha(z_score: Union[float, np.ndarray]) -> np.ndarray:
    """Recover the alpha (significance level) given by `alpha = 2 * (1 - normal_cdf(z_score))`.
    Args:
        z_score: z_score aka `z`
    Returns:
        alpha: significance level
    """
    return 2 * (1 - stats.norm.cdf(z_score))

def plot_empirical_z_score(
    axes: Axes,
    nominal_z_scores: np.ndarray,
    z_mean: np.ndarray,
    z_interval: np.ndarray,
    mean_color: str = "black",
    interval_color: str = "0.8",
    diagonal_color: str = "darkgreen",
    sigma_color: str = "red",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = r"Empirical coverage [$z_p$]",
    ylabel: Optional[str] = r"Nominal credibility [$z_p$]",
    diagonal_text: bool = False,
    fsize: int = 12,
    lw_horizonal=1,
    lw_vertical=1,
    lw_empirical=1,
    lw_diagonal=1
) -> Axes:
    """target a particular matplotlib Axes and produce an empirical coverage test plot with Jeffrey's interval
    Args:
        axes: matplotlib axes
        nominal_z_scores: sorted array of nominal z-scores
        z_mean: empirical mean of z-score estimate using a binominal distribution
        z_interval: jeffrey's interval of z-score estimate
        mean_color: color of the mean line.
        interval_color: color of the interval, floats are grey.
        diagonal_color: color of the diagonal, nominal z-score.
        sigma_color: color of the vertical and horizontal sigma lines.
        xlim: force xlim
        ylim: force ylim
        xlabel: set xlabel
        ylabel: set ylabel
        diagonal_text: turns on semantic description of above / below diagonal
    Returns:
        the matplotlib axes given
    """
    lower = z_interval[:, 0]
    upper = z_interval[:, 1]
    assert np.all(lower <= upper), "the lower interval must be <= the upper interval."
    upper = np.where(upper == np.inf, 100.0, upper)

    # empirical lines & interval
    axes.plot(nominal_z_scores, z_mean, color=mean_color, lw=lw_empirical)
    axes.fill_between(nominal_z_scores, lower, upper, color=interval_color)

    # diagonal line
    max_z_score = np.max(nominal_z_scores)
    axes.plot([0, max_z_score], [0, max_z_score], "--", color=diagonal_color, lw=lw_diagonal)

    # horizontal and vertical lines, vertical are the "truth", horizontal are empirical
    for i_sigma in range(1, int(max_z_score) + 1):
        empirical_i_sigma = np.interp(i_sigma, nominal_z_scores, z_mean)
        if empirical_i_sigma != np.inf:  # when the vertical line intersects z_mean
            # Horizontal line
            axes.plot(
                [0, i_sigma],
                [empirical_i_sigma, empirical_i_sigma],
                ":",
                color=sigma_color,
                lw=lw_horizonal
            )
            # horizontal text
            c = 1 - get_alpha(empirical_i_sigma)
            axes.text(0.15, empirical_i_sigma + 0.05, (r"$%.2f$" % (c * 100)) + "$\%$", fontsize=fsize)
            # vertical line
            axes.plot(
                [i_sigma, i_sigma], [0, empirical_i_sigma], ":", color=sigma_color, lw=lw_vertical
            )
            # vertical text
            c = 1 - get_alpha(i_sigma)
            axes.text(i_sigma, 0.15, (r"$%.2f$" % (c * 100)) + "$\%$", rotation=-90, fontsize=fsize)
        else:  # when the vertical line fails to intersect z_mean
            pass

    # set labels
    axes.set_ylabel(xlabel)
    axes.set_xlabel(ylabel)

    # Add the semantic meaning of being above / below diagonal
    if diagonal_text:
        raise NotImplementedError("must add rotation description")

    # set limits
    if xlim is None:
        axes.set_xlim([0, max_z_score])
    else:
        axes.set_xlim(xlim)

    if ylim is None:
        axes.set_ylim([0, max_z_score + np.round(0.15 * max_z_score, 1)])
    else:
        axes.set_ylim(ylim)
    return axes

def coveragePlot(coverage_samples, labels, cmap="greys_r"):
    fig, ax = plt.subplots(6, 6, figsize=(10, 10), sharex=True, sharey=True)
    cn = 0
    lower_lim = 1
    for i in range(0, 5):     
        for j in range(lower_lim, 6):
            cov = sl.estimate_coverage(coverage_samples, coverage_samples[1].parnames[cn].tolist())
            plot_empirical_z_score(ax[j, i], cov[:, 0], cov[:, 1], cov[:, 2:], xlabel="", ylabel="",
                                   mean_color=cmap(0.999), interval_color=cmap(0.2), sigma_color="grey",
                                   diagonal_color="g", fsize=6, lw_horizonal=0.5, lw_vertical=0.5,
                                   lw_diagonal=0.5)
            cn += 1     
        lower_lim+=1      

    for i in range(6):
        for j in range(6):
            ax[i, j].tick_params(axis='both', which='major', labelsize=8, length=3)
            ax[i, j].tick_params(axis='both', which='minor', labelsize=8, length=2)
            if j > i:
                ax[i, j].set_axis_off()
            ax[i, j].set_xticks([0, 1, 2, 3])
            if i == j:
                cov = sl.estimate_coverage(coverage_samples, "z[%d]"%i)
                plot_empirical_z_score(ax[i, j], cov[:, 0], cov[:, 1], cov[:, 2:], xlabel="", ylabel="",
                                       mean_color=cmap(0.999), interval_color=cmap(0.2), sigma_color="grey",
                                       diagonal_color="g", fsize=6, lw_horizonal=0.5, lw_vertical=0.5,
                                       lw_diagonal=0.5)

    for i in range(len(labels)):
        ax[i, 0].set_ylabel(labels[i], fontsize=12)
        ax[-1, i].set_xlabel(labels[i], fontsize=12)
        ax[i, 0].get_yaxis().set_label_coords(-0.25, 0.5)
        ax[-1, i].get_xaxis().set_label_coords(0.5, -0.25)
    ax[0, 0].set_ylabel("")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.align_xlabels()
    fig.align_ylabels()
    
def plotIonzHist(reds, mode, min_s1, max_s1, min_s2, max_s2, obs0):
    fig, ax1 = plt.subplots()
    def zTonu(z):
        return 1420./(1.+np.array(z))

    def nuToz(nu):
        return 1420./nu - 1.

    mode = np.array(mode)
    errors_1 = np.array([min_s1, max_s1])
    errors_2 = np.array([min_s2, max_s2])

    ax1.errorbar(1420./(1.+np.array(reds)), mode, yerr=errors_2, fmt=".", capsize=0, ecolor="cyan",
                 lw=1.5, label=r"$2\sigma$")
    ax1.errorbar(1420./(1.+np.array(reds)), mode, yerr=errors_1, fmt=".", capsize=0, color="k",
                 ecolor="darkturquoise", lw=4, label=r"$1\sigma$")
    ax1.plot(1420./(1.+np.array(reds)), obs0['xHI'], c="coral", ls="--")
    ax2 = ax1.secondary_xaxis('top', functions=(nuToz, zTonu))

    ax2.set_xticks(reds)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax1.set_ylim(0, 1.05)
    ax1.legend()
    plt.show()