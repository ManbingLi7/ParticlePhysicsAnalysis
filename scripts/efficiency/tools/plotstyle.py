figure.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
# draw data and fitted line
plt.errorbar(data_x, data_y, data_yerr, fmt="ok", label="data")
plt.plot(data_x, line(data_x, *m.values), label="fit")

# display legend with some fit info
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel("x")
plt.ylabel("y");



legend_elements = [
    Line2D([0], [0], marker='o', color='b', label='Circle'),
    Line2D([0], [0], marker='s', color='r', label='Square')
]

#with uproot.open(filenames) as massfile:
        #hist2d_mass_energy = Histogram.from_file(massfile, f"Be_{detector}_mass_ciemat.npy")
        #print(hist2d_mass_energy.values())
        #print(hist2d_mass_energy.errors())
        #print(hist2d_mass_energy.axes[0].edges())
        #print(hist2d_mass_energy.axes[1].edges())
        
plt.axhline(y=1.0, color='red', linestyle='--', label='y = 1.0')

ax1.text(0.9, 0.95, f"{dec}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")


    ax2.grid(linewidth=1)
    ax2.legend(fontsize=FONTSIZE, loc="lower right")
    ax1.set_ylim([0.95, 1.05])
    ax2.set_ylim([0.99, 1.007])
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim([1.9, 1300])
    ax2.set_xticks([2, 5,  10, 30,  80,  300, 1000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yticklabels()[0].set_visible(False)
    ax1.sharex(ax2)
    
    ax1.legend(loc='lower right', fontsize=FONTSIZE)
    ax2.set_xlabel(RIG_XLABEL)
    ax2.set_ylabel("this/J.W")
    ax1.text(0.05, 0.95, "BZ", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
    ax2.text(0.05, 0.95, "Difference within 0.3%", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="red", weight='bold')
    set_plot_defaultstyle(ax1)
    set_plot_defaultstyle(ax2)
    plt.subplots_adjust(hspace=.0)
    savefig_tofile(figure, args.resultdir, f"hist_compare_jw_bz_effcor", show=True)



