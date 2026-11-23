"""Plot mean chirp fringes, single-sweep fluctuations, and their mean shift."""

from pathlib import Path

import numpy as np


def plot_chirp_comparison(rows, out_dir):
    """Write a PNG preview and vector PDF from the per-chirp readout table.

    P2 columns are fractions. The residuals and the mean shift are plotted
    in percentage points; the two shaded bands have different meanings.
    """
    from matplotlib import font_manager, pyplot as plt

    def column(name):
        return np.asarray([r[name] for r in rows], dtype=float)

    x = column("chirp_Hz_s") / 1e6
    baseline = 100 * column("baseline_mean_P2")
    combined = 100 * column("combined_mean_P2")
    fonts = {entry.name for entry in font_manager.fontManager.ttflist}
    preferred = ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
                 "PingFang SC", "SimHei", "WenQuanYi Zen Hei")
    chinese_font = next((font for font in preferred if font in fonts), None)

    def label(zh, en):
        return zh if chinese_font else en

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": [chinese_font, "DejaVu Sans"] if chinese_font else ["DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
    base_color, noisy_color = "#26568b", "#af501c"
    base_label = label("仅探测噪声", "Detection only")
    noisy_label = label("探测及四项技术噪声", "Detection + four technical terms")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig, (mean, sweep, difference) = plt.subplots(
            3, sharex=True, figsize=(8.2, 7.0), gridspec_kw={"height_ratios": [1.65, 1., 1.]})
        fig.subplots_adjust(left=.13, right=.97, top=.91, bottom=.09, hspace=.44)
        fig.suptitle(label("三脉冲 Raman 干涉的扫啁啾条纹", "Three-pulse Raman chirp fringes"),
                     x=.13, y=.991, ha="left", va="top", fontsize=14)

        mean.plot(x, baseline, color=base_color, lw=2.6, label=base_label)
        mean.plot(x, combined, color=noisy_color, lw=1.6, ls=(0, (4, 2)), label=noisy_label)
        mean.plot(x, 100 * column("combined_example_P2"), ".", ms=3.5,
                  color=noisy_color, alpha=.7, label=label("全噪声单扫", "Full-noise sweep"))
        mean.set_ylabel(label("归一化荧光读数 (%)", "Normalized fluorescence (%)"))

        sweep.axhline(0, color="#7b8287", lw=.65, zorder=0)
        sweep.fill_between(x, 100 * column("combined_q025") - combined,
                           100 * column("combined_q975") - combined,
                           color=noisy_color, alpha=.16, linewidth=0,
                           label=label("全噪声单门 95% 区间", "Full-noise 95% gate interval"))
        sweep.plot(x, 100 * column("baseline_example_P2") - baseline, "o",
                   ms=2.8, color=base_color, label=label("仅探测", "Detection only"))
        sweep.plot(x, 100 * column("combined_example_P2") - combined, "x",
                   ms=3.0, mew=.85, color=noisy_color, label=label("全噪声", "Full noise"))
        sweep.set_ylabel(label("读数偏差（百分点）", "Deviation (pp)"))

        shift = combined - baseline
        se = 100 * column("combined_mean_se_P2")
        difference.axhline(0, color="#7b8287", lw=.7, ls="--", zorder=0)
        difference.fill_between(x, shift - se, shift + se, color=noisy_color, alpha=.22,
                                linewidth=0, label=label("±1 蒙特卡洛标准误", "±1 Monte Carlo SE"))
        difference.plot(x, shift, color=noisy_color, lw=1.35)
        difference.set_ylabel(label("均值差（百分点）", "Mean shift (pp)"))
        difference.set_xlabel(label("双光子频差啁啾率 α (MHz/s)", "Two-photon chirp rate α (MHz/s)"))

        panel_labels = [
            label("a   平均干涉条纹", "a   Mean interference fringes"),
            label("b   单次扫描相对各自均值的偏差", "b   Single-sweep deviations from each mean"),
            label("c   加入技术噪声后的平均读数变化", "c   Mean shift after adding technical noise"),
        ]
        for ax, title in zip((mean, sweep, difference), panel_labels):
            ax.set_title(title, loc="left", fontsize=9.5, pad=7)
            ax.legend(loc="lower right", bbox_to_anchor=(1, 1.016), ncol=3,
                      frameon=False, borderaxespad=0, handlelength=1.5,
                      columnspacing=1.1, handletextpad=.4)
            ax.grid(axis="y", color="#d9e1e5", lw=.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=3, width=.8)
            ax.margins(x=.012, y=.12)
        fig.savefig(out_dir / "chirp_noise_comparison.png", dpi=300, facecolor="white")
        fig.savefig(out_dir / "chirp_noise_comparison.pdf", facecolor="white")
        plt.close(fig)
