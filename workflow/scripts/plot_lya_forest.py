import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import lya_forest

# get values injected to global by snakemake
# pylint: disable=undefined-variable
forest_spectra = snakemake.output[0]
W_vs_NHI = snakemake.output[1]
du_tau_hist = snakemake.output[2]
config = snakemake.config
# pylint: enable=undefined-variable

# update the rcParams
plt.rcParams.update(config["plotting"]["rcParams"])

# create the Lyman-alpha Forest degrader
lya_config = config["catalog"]["lya_extinction"]
del lya_config["seed"]

lya = lya_forest.LyAForestExtinction(**lya_config)

# simulate 10,000 lines of sight
lines_of_sight = lya._simulate_lines_of_sight(range(10_000), 0)


# ------------------------------------
# Plots of Lyman-alpha Forest Spectra
# ------------------------------------

fig, axes = plt.subplots(5, 1, figsize=(3.5, 5), constrained_layout=True)

# plot the u band response function at the top
axes[0].plot(lya._u_wave, lya._u_R(lya._u_wave))
axes[0].text(0.20, 0.65, "$R_u(\lambda)$", transform=axes[0].transAxes)

# plot the absorption spectrum for a few galaxies
for i, ax in enumerate(axes[1:]):

    # get the redshifts and eqivalent widths
    zs = lines_of_sight[i]["z"]
    eqWs = lines_of_sight[i]["eqW"]

    for z, W in zip(zs, eqWs):
        # calculate the location of the line
        wavelen = (1 + z) * 1215.67

        # stretch the width due to redshift
        # I also doubled the width to make thinner lines more visible
        W = W * (1 + z)

        # plot the absorption bar
        ax.axvspan(
            wavelen - W / 2,
            wavelen + W / 2,
            0.02,
            0.98,
        )

    # print stats in the title
    N_clouds = lines_of_sight[i]["N_clouds"]
    u_decr = lines_of_sight[i]["u_decr"]
    tau_u = lines_of_sight[i]["tau_u"]
    ax.set_title(
        (
            f"$N_{{\mathrm{{clouds}}}}$ = {N_clouds},  "
            f"$\\tau_{{u}}$ = {tau_u:.3f},  "
            f"$\Delta u$ = {u_decr:.3f} mag"
        ),
        # fontsize = 10,
        pad=2,
    )

# set axis properties for all the axes
for ax in axes:
    ax.set(
        xlim=(lya._u_wave.min(), lya._u_wave.max()),
        xticks=np.arange(3200, 4200, 200),
        yticks=[],
    )
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

# remove x tick labels for the top plots
for ax in axes[:-1]:
    ax.set(
        xticklabels=[],
    )

# add x axis label to the bottom plot
axes[-1].set(xlabel="Wavelength [$\AA$]")

# add a redshift axis below the plot
z_ax = axes[-1].twiny()
z_ax.xaxis.set_ticks_position("bottom")
z_ax.xaxis.set_label_position("bottom")
z_ax.spines["bottom"].set_position(("axes", -1))
z_ax.spines["left"].set_visible(False)
z_ax.spines["right"].set_visible(False)
z_ax.spines["top"].set_visible(False)

# set the redshifts marked on this axis
zs = np.array(
    [lya._u_wave.min() / 1215.67 - 1, 1.8, 2, 2.2, lya._u_wave.max() / 1215.67 - 1]
)

# set redshift axis properties
xticks = 1215.67 * (1 + zs)
xticklabels = [f"{z:.1f}" for z in zs]
z_ax.set(
    xlim=(lya._u_wave.min(), lya._u_wave.max()),
    xticks=xticks,
    xticklabels=xticklabels,
    xlabel="Redshift of absorber",
)

fig.savefig(forest_spectra, dpi=300)


# ----------------
# Plot eqW vs NHI
# ----------------

fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)

logNHIs = []
logWs = []
for los in lines_of_sight.values():
    logNHIs += list(np.log10(los["NHI"]))
    logWs += list(np.log10(los["eqW"]))

ax.hist2d(logNHIs, logWs, bins=500, cmap="Blues", norm=mpl.colors.LogNorm())

# plot the arrows and values of b
ax.annotate(
    "",
    xy=(15.5, -0.45),
    xytext=(15.2, 0.06),
    arrowprops=dict(arrowstyle="<-", color="k"),
)
ax.text(15.6, -0.44, "$b=24$", va="top", ha="center")
ax.text(15.25, 0.06, f"$b={lya._presampled_bs.max():.0f}$", va="bottom", ha="center")

# set the axis labels
ax.set(
    xlabel="$\log_{10}(N_{HI} ~ [\mathrm{cm}^{-2}])$",
    ylabel="$\log_{10}(W ~ [\AA])$",
    yticks=[-1.5, -1, -0.5, 0],
    xticks=[13, 14, 15, 16],
)

fig.savefig(W_vs_NHI, dpi=300)


# -----------------------------------------------------
# Plot distribution of u decrements and optical depths
# -----------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), constrained_layout=True)

u_decrs = []
taus = []
for los in lines_of_sight.values():
    u_decrs.append(los["u_decr"])
    taus.append(los["tau_u"])

names = ["\Delta u", "\\tau_u"]
units = [" [mag]", ""]
for ax, vals, name, unit in zip(axes, [u_decrs, taus], names, units):

    ax.hist(vals, bins="fd", density=True, histtype="stepfilled")

    mean = np.mean(vals)
    std = np.std(vals)
    ax.set(
        xlabel=f"${name}${unit}",
        title=f"$\mu_{{{name}}} = {mean:.2f}, ~ \sigma_{{{name}}} = {std:.2f}$",
        yticks=[],
        xlim=(mean - 4 * std, mean + 4 * std),
    )

fig.savefig(du_tau_hist, dpi=300)
