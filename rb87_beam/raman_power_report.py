"""Export the multilevel optical-power calibration of a frozen Raman design.

Run from the repository root: python -m rb87_beam.raman_power_report.
The exporter maps an existing effective coupling onto explicit optical inputs;
it does not rerun the optimization or claim a new scattering-inclusive fringe.
"""

import argparse
import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .physics import raman_peak_frequency, raman_optical_frequency_offset
from .raman_calibration import (STECK_D2_CG_REDUCED_CM, D2_CYCLING_DIPOLE_CM,
                                calibrate_powers, solve_powers)
from .states import G_CLOCK, E_CLOCK


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table(headers, rows):
    return ("<div class='table'><table><thead><tr>" +
            "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers) +
            "</tr></thead><tbody>" + "".join("<tr>" +
            "".join("<td>" + html.escape(str(x)) + "</td>" for x in row) +
            "</tr>" for row in rows) + "</tbody></table></div>")


def _number(value):
    return "%.9g" % float(value)


def _plot(rows, powers, output, ratio_rule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = np.array([r["total_power_mW"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    selected_label = "DLS-cancelling ratio" if ratio_rule == "cancel_light_shift" else "Selected power ratio"
    for prefix, label, style in (("selected", selected_label, "-"),
                                  ("equal", "Equal powers (1:1)", "--")):
        axes[0].plot(total, [r[prefix + "_peak_rabi_kHz"] for r in rows], style, label=label, lw=2)
        axes[1].plot(total, [r[prefix + "_differential_light_shift_kHz"] for r in rows], style, label=label, lw=2)
    for row, label in ((powers[0], "First / third"), (powers[1], "Middle")):
        axes[0].plot(row["total_power_mW"], row["peak_rabi_kHz"], "o", color="#a63a24")
        axes[0].annotate(label, (row["total_power_mW"], row["peak_rabi_kHz"]),
                         xytext=(6, -15), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="Total two-color power at atoms (mW)", ylabel="Effective Rabi frequency / 2pi (kHz)")
    axes[1].set(xlabel="Total two-color power at atoms (mW)", ylabel="Differential AC light shift (kHz)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("87Rb D2 clock Raman transition: coherent hyperfine-path power calibration", fontsize=13)
    fig.savefig(output / "power_curve.png", dpi=180)
    plt.close(fig)


def generate_report(config_path, output_dir, reference_Fprime=1,
                    power_ratio="cancel_light_shift", detuning_Hz=None):
    """Generate numerical outputs and an opt-in, directly runnable power config."""
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    original_bytes = config_path.read_bytes()
    config = SimulationConfig.from_dict(json.loads(original_bytes.decode("utf-8-sig")))
    if config.raman.envelope != "gaussian":
        raise ValueError("Actual-power exporter currently requires matched Gaussian beams")
    target = float(raman_peak_frequency(config.raman))
    if target <= 0:
        raise ValueError("The frozen design must have a nonzero effective Raman coupling")
    if abs(config.raman.differential_light_shift_Hz) > 1e-9:
        raise ValueError("This exporter maps a zero-residual-light-shift design; do not silently remove an existing residual shift")
    r = config.raman
    settings = dict(waist_y_m=r.waist_y_m, waist_z_m=r.waist_z_m,
        detuning_Hz=r.single_photon_detuning_Hz if detuning_Hz is None else detuning_Hz,
        reference_Fprime=reference_Fprime, polarization1="sigma_plus", polarization2="sigma_plus",
        two_photon_detuning_Hz=raman_optical_frequency_offset(r))
    selected = solve_powers(target, **settings, power_ratio=power_ratio)
    equal = solve_powers(target, **settings, power_ratio=1.)
    powers, zone_calibrations = [], []
    for zone, factor in enumerate((1., 2., 1.), 1):
        result = calibrate_powers(factor * selected["power1_W"], factor * selected["power2_W"], **settings)
        zone_calibrations.append(result)
        powers.append(dict(zone=zone, pulse="pi" if zone == 2 else "pi/2", coupling_scale=factor,
            target_rabi_kHz=factor * target / 1e3, peak_rabi_kHz=result["peak_rabi_Hz"] / 1e3,
            reference_area_pi=(2 * result["peak_rabi_Hz"] * r.waist_z_m * np.sqrt(np.pi / 2)
                               / r.reference_vz_m_s),
            truncated_reference_area_pi=(2 * result["peak_rabi_Hz"] * r.waist_z_m * np.sqrt(np.pi / 2)
                / r.reference_vz_m_s * math.erf(np.sqrt(2) * r.envelope_half_widths)),
            beam1_high_frequency_power_mW=result["power1_W"] * 1e3,
            beam2_low_frequency_power_mW=result["power2_W"] * 1e3,
            total_power_mW=(result["power1_W"] + result["power2_W"]) * 1e3,
            power_ratio_high_over_low=selected["power_ratio"],
            differential_light_shift_Hz=result["differential_light_shift_Hz"],
            clock_F1_m0_peak_scattering_rate_per_s=result["ground_scattering_rates_per_s"][G_CLOCK],
            clock_F2_m0_peak_scattering_rate_per_s=result["ground_scattering_rates_per_s"][E_CLOCK]))
    calibrated = copy.deepcopy(config)
    pr = calibrated.raman
    pr.coupling_model = "rb87_d2_power"
    pr.beam1_power_W, pr.beam2_power_W = selected["power1_W"], selected["power2_W"]
    pr.peak_rabi_Hz = 0.
    pr.single_photon_detuning_Hz = settings["detuning_Hz"]
    pr.detuning_reference_Fprime = reference_Fprime
    pr.polarization1 = pr.polarization2 = "sigma_plus"
    pr.quantization_axis = tuple(pr.k_direction)
    pr.spin_transport = "adiabatic_from_z"
    pr.differential_light_shift_Hz = 0.
    # The legacy six-pi demonstration can exceed the far-detuned bound when
    # converted into actual optical fields. Use an explicitly bounded example.
    calibrated.rabi_scan.area_min_pi = 0.
    calibrated.rabi_scan.area_max_pi = 2.
    calibrated.rabi_scan.transit_area_pi = 2.
    calibrated.validate()
    if not np.isclose(raman_peak_frequency(pr), target, rtol=1e-11, atol=1e-7):
        raise RuntimeError("Exported actual-power config does not reproduce its target coupling")
    ratio = selected["power_ratio"]
    curve = []
    for total_mW in np.linspace(0., powers[1]["total_power_mW"] * 1.2, 121):
        total_W = total_mW * 1e-3
        a = calibrate_powers(total_W * ratio / (1 + ratio), total_W / (1 + ratio), **settings)
        b = calibrate_powers(total_W / 2, total_W / 2, **settings)
        curve.append(dict(total_power_mW=float(total_mW),
            selected_beam1_power_mW=a["power1_W"] * 1e3, selected_beam2_power_mW=a["power2_W"] * 1e3,
            selected_peak_rabi_kHz=a["peak_rabi_Hz"] / 1e3,
            selected_differential_light_shift_kHz=a["differential_light_shift_Hz"] / 1e3,
            equal_beam1_power_mW=b["power1_W"] * 1e3, equal_beam2_power_mW=b["power2_W"] * 1e3,
            equal_peak_rabi_kHz=b["peak_rabi_Hz"] / 1e3,
            equal_differential_light_shift_kHz=b["differential_light_shift_Hz"] / 1e3))
    data = dict(input_config=str(config_path), input_config_sha256=hashlib.sha256(original_bytes).hexdigest(),
        input_effective_peak_rabi_Hz=target, settings=settings,
        configured_residual_two_photon_detuning_Hz=r.two_photon_detuning_Hz,
        reference_vz_m_s=r.reference_vz_m_s,
        selected=selected, equal_power_comparison=equal, zone_calibrations=zone_calibrations,
        powers=powers, power_driven_config=calibrated.to_dict(),
        units=dict(power="W in config and calibration API; mW in tables and CSV", frequencies="Hz except explicit kHz", waists="1/e^2 intensity radii in m"),
        assumptions=dict(
            detuning_reference="beam 1 frequency minus F=1 -> Fprime=%d zero-field optical transition; positive means blue" % reference_Fprime,
            reference_is_new_explicit_assumption=True, beam1="higher optical frequency, +k, addresses F=1",
            beam2="lower optical frequency, -k, addresses F=2",
            quantization_axis=list(pr.quantization_axis), polarization="same sigma_plus relative to the common quantization axis; opposite propagation-relative helicities",
            spin_transport="ideal nonzero-field adiabatic rotation from preparation z to Raman axis; populations preserved, not reset",
            power_location="individual frequency/direction component power at atoms, excluding optical transmission and unused sidebands",
            pulse_scaling="same waists in all zones; both powers double in the middle zone",
            pulse_area="pulse labels denote nominal splitter/mirror roles; reference_area_pi integrates the Gaussian coupling at reference_vz, not an enforced pi/2 or pi rotation",
            light_shift="both optical frequencies coupled to all D2 excited hyperfine states, including wrong legs",
            zeeman="weak guide field only defines axis; Zeeman shifts omitted in optical denominators",
            scattering="peak spontaneous-rate diagnostics only; not dissipative Raman trajectories",
            rabi_example="reference-area scan 0 to 2 pi, single-transit example 2 pi; each physical power must remain in the far-detuned regime",
            scope="calibration of the frozen effective design, not a new scattering-inclusive optimization"),
        constants=dict(steck_D2_CG_reduced_dipole_C_m=STECK_D2_CG_REDUCED_CM,
                       D2_cycling_dipole_C_m=D2_CYCLING_DIPOLE_CM),
        sources=["https://steck.us/alkalidata/rubidium87numbers.pdf",
                 "https://jila.colorado.edu/~junye/yelabsOLD/pubs/scienceArticles/1990-1998/sArticle_1996_08_RbHyperfine.pdf",
                 "https://arxiv.org/pdf/1605.05230",
                 "https://web.stanford.edu/group/kasevich/cgi-bin/wordpress/wp-content/uploads/2022/04/Y-WU_2021_PhD_Thesis.pdf"])
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "calibration.json", data)
    calibrated.save(output / "config_power_driven.json")
    _csv(output / "powers.csv", powers)
    _csv(output / "power_curve.csv", curve)
    _plot(curve, powers, output, selected["power_ratio_rule"])
    _write_html(data, output)
    return data


def _verification_html(data, output):
    """Include existing paired replay evidence, without launching simulations."""
    path = output / "verification.json"
    if not path.exists():
        return ""
    verification = json.loads(path.read_text(encoding="utf-8-sig"))
    scaling = verification["power_scaling"]
    selected = data["selected"]
    power_config_sha256 = hashlib.sha256(json.dumps(data["power_driven_config"],
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    coefficient = selected["peak_rabi_Hz"] / 1e3 / np.sqrt(
        selected["power1_W"] * selected["power2_W"] * 1e6)
    matches = (verification.get("power_config_sha256") == power_config_sha256
        and verification.get("input_effective_config_sha256") == data["input_config_sha256"]
        and np.isclose(scaling["target_rabi_Hz"], selected["peak_rabi_Hz"], rtol=1e-10)
        and np.isclose(scaling["coefficient_kHz_per_sqrt_mW_product"], coefficient, rtol=1e-10)
        and np.isclose(scaling["base_differential_light_shift_Hz"],
                       selected["differential_light_shift_Hz"], rtol=1e-10, atol=1e-6))
    parts = ["<h2>已有原子样本上的对照验证</h2>"]
    if not matches:
        parts.append("<p>输出目录已有验证记录，但其原有效配置或实际功率配置的摘要与本次设置不匹配，旧验证不适用于本次报告；请按本次配置重新验证。</p>")
        return "\n".join(parts)
    else:
        parts.append("<p>固定本报告的光斑、失谐、偏振和参考光频差时，功率关系为 <b>f<sub>eff</sub>[kHz] = %s √(P1[mW] P2[mW])</b>。功率反算频率的相对误差为 %s；两束功率同时加倍时，拉比频率乘 %s。仅将高光频束 P1 加倍时，拉比频率乘 %s，同时差分光移变为 %s kHz，不能把这两种调节视为相同操作。</p>" %
            (_number(coefficient), _number(scaling["roundtrip_relative_error"]),
             _number(scaling["both_powers_doubled_rabi_ratio"]),
             _number(scaling["beam1_only_doubled_rabi_ratio"]),
             _number(scaling["beam1_only_doubled_differential_light_shift_Hz"] / 1e3)))
        replay = verification["paired_replay"]
        labels = {"phase": "附加相位扫描", "rabi": "Rabi振荡扫描", "chirp": "啁啾率扫描"}
        rows = []
        for experiment in ("phase", "rabi", "chirp"):
            record = replay[experiment]
            rows.append([labels[experiment], record["atoms"], record["points"],
                _number(record["max_abs_per_atom_P2_difference"]),
                _number(record["max_abs_expected_photon_P2_difference"]),
                "一致" if record["identical_positions_velocities_and_density"] else "不一致"])
        parts.append("<p>从同一个已完成光泵浦的原子检查点重放，将实际功率模式与原有效频率模式逐点比较；每项使用相同的 %s 个原子，束流归一化通量为 %s s⁻¹。比较的是逐原子的 F=2 概率和期望荧光归一化读数，未用独立光子噪声样本掩盖数值差异。</p>" %
            (_number(replay["phase"]["atoms"]), _number(replay["phase"]["flux_per_s"])))
        parts.append(_table(["扫描", "原子数", "扫描点数", "逐原子概率最大绝对差", "期望荧光P2最大绝对差", "初始位置、速度和态密度矩阵"], rows))
        if "unit_tests" in verification:
            parts.append("<p>验证记录中的自动测试通过 %s 项。上述对照验证了功率换算与既有相干传播的一致性；同一批原子的重放不等同于扩大样本量的收敛检验，也没有增加散射退相干。</p>" %
                         _number(verification["unit_tests"]["passed"]))
    links = [("verification.json", "完整验证记录"),
             ("replay/rabi/rabi.png", "Rabi振荡图"),
             ("replay/chirp/chirp.png", "变啁啾三脉冲条纹"),
             ("replay/phase/overview.png", "附加相位条纹")]
    parts.append("<p>" + " · ".join("<a href='%s'>%s</a>" %
        (html.escape(name, quote=True), html.escape(label)) for name, label in links
        if (output / name).is_file()) + "</p>")
    return "\n".join(parts)


def _write_html(data, output):
    settings, selected, equal = data["settings"], data["selected"], data["equal_power_comparison"]
    ratio_rule = selected["power_ratio_rule"]
    rule = "抵消两个钟态的差分 AC 光移" if ratio_rule == "cancel_light_shift" else "使用指定的两色功率比"
    powers = data["powers"]
    cancellation_note = ("所选功率比抵消差分光移，并不抵消两个基态的共同光移或自发散射。"
        if ratio_rule == "cancel_light_shift" else
        "本次使用手动指定的功率比，其差分光移如上表所示，并未自动抵消。")
    roles = {1: "分束", 2: "反射", 3: "合束"}
    rows = [[r["zone"], roles[r["zone"]], _number(r["target_rabi_kHz"]),
             _number(r["reference_area_pi"]),
             _number(r["beam1_high_frequency_power_mW"]), _number(r["beam2_low_frequency_power_mW"]),
             _number(r["total_power_mW"])] for r in powers]
    compare = []
    for name, result in (("本次功率比", selected), ("等功率 1:1 对照", equal)):
        compare.append([name, _number(result["power_ratio"]),
            _number(1e3 * result["power1_W"]), _number(1e3 * result["power2_W"]),
            _number(result["peak_rabi_Hz"] / 1e3), _number(result["differential_light_shift_Hz"] / 1e3)])
    text = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Rb87：Raman实际功率与多能级有效拉比频率标定</title>",
        "<style>body{font-family:'Microsoft YaHei',system-ui,sans-serif;color:#183040;max-width:1200px;margin:36px auto;padding:0 24px;line-height:1.8;background:#fbfcfd}h1{font-size:29px;line-height:1.4}h2{margin-top:30px;font-size:21px}.lead{padding:16px 22px;background:#eaf4f6;border-left:4px solid #247c91}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dce5ea;text-align:left;vertical-align:top}th{background:#eef4f7}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f6;padding:16px;font-size:13px}.scope{padding:14px 20px;background:#f2f5f8}.small{font-size:13px;color:#526c7a}a{color:#176987}</style></head><body>",
        "<h1>Rb87：Raman实际光功率标定</h1>",
        "<p class='lead'>在本页明确的多能级、偏振和失谐假设下，第一、第三光区所需高光频束功率为 <b>%.6f mW</b>，低光频束为 <b>%.6f mW</b>；中间光区两束各加倍。功率均指原子处的单个光频分量。</p>" %
        (selected["power1_W"] * 1e3, selected["power2_W"] * 1e3),
        "<p>这是将已选定的有效 Raman 耦合换算为明确的光学输入，不是重新搜索一套最优条纹参数。</p>",
        "<h2>三光区所需的两束光功率</h2>",
        _table(["光区", "功能", "目标 Ωeff/(2π) / kHz", "参考面积 / π", "高光频束 P1 / mW", "低光频束 P2 / mW", "两色合计 / mW"], rows),
        "<p>两束光在三个区域都使用相同的椭圆高斯光斑：w<sub>y</sub> = %s mm，w<sub>z</sub> = %s μm，均为光强 1/e² 半径。两色功率比 P1/P2 = <b>%s</b>，选择规则是%s。第二光区的两束功率各乘2，使 Ωeff 翻倍，光移和峰值散射率也按功率同比缩放。</p>" %
        (_number(settings["waist_y_m"] * 1e3), _number(settings["waist_z_m"] * 1e6), _number(selected["power_ratio"]), rule),
        "<p>参考束速为 %s m/s，表中面积按 ∫Ωeff(t)dt/π = 2 f<sub>eff</sub>w<sub>z</sub>√(π/2)/v<sub>ref</sub> 计算，依次为 %sπ、%sπ、%sπ。因此“分束—反射—合束”表示光区功能，当前优化没有强制面积严格等于 π/2—π—π/2。有限光区截断修正另存 truncated_reference_area_pi；真实原子的速度、横向位置及失谐还会改变其响应。</p>" %
        (_number(data["reference_vz_m_s"]), *[_number(r["reference_area_pi"]) for r in powers]),
        "<h2>量子轴、偏振和频率约定</h2>",
        "<p>本次假设 Raman 区的量子化轴与高光频束的传播方向一致，当前方向为 (%s)；默认几何为高光频束沿 +x、低光频束沿 −x。两束光相对同一个 Raman 量子轴均为 σ⁺，因此相对各自传播方向的圆偏振手性相反。制备区磁场原先沿 z，模型显式假设非零导引磁场将量子轴从 z 绝热转向 Raman 轴，保留各 |F,m<sub>F</sub>⟩ 人口；这里没有将原子重置到纯 m<sub>F</sub>=0，也没有计算实际输运损失和附加 Zeeman 相位。</p>" %
        ", ".join(_number(x) for x in data["assumptions"]["quantization_axis"]),
        "<p>将已有单光子失谐数值明确解释为高光频束相对 <b>F=1 → F′=%d</b> 零场跃迁的失谐：<b>%s MHz</b>，正号表示蓝失谐。这一参考线是本次新增的显式假设，旧配置没有指定它。低光频束频率比高光频束低 ν<sub>hfs</sub> + δ，其中用于载频分母的本次 δ = %s kHz。光学分母采用一致的零场超精细能级；弱导引磁场只用于定义量子轴，未加入其 Zeeman 修正。</p>" %
        (settings["reference_Fprime"], _number(settings["detuning_Hz"] / 1e6), _number(settings["two_photon_detuning_Hz"] / 1e3)),
        "<p>配置中的 two_photon_detuning_Hz = %s kHz 是相对“钟频＋两光子反冲＋参考 Doppler”共振条件的剩余失谐；载频标定已加入这些参考偏置，因此上面的 δ 不应直接当作该配置字段。每个原子实际速度导致的单光子 Doppler、单光子反冲及 Zeeman 对 GHz 光学分母的小修正仍未计算。</p>" % _number(data["configured_residual_two_photon_detuning_Hz"] / 1e3),
        "<p>功率是原子处的每个光频、传播方向分量，不是激光器总输出功率。光路透过率、AOM/EOM效率、未参与 Raman 的边带和多余反向分量需要另行计入。80 μW 的态制备泵浦功率属于另一组光场。</p>",
        "<h2>多能级换算与等功率对照</h2>",
        "<pre>I0,j = 2 Pj / (π wy wz),    Ej = sqrt(2 I0,j / (ε0 c))\nΩj,eg = Ej (d_eg · εj) / ℏ\nΩeff = Σe [Ω1,e1 Ω2,e2* / 4] [1/Δ1,e1 + 1/Δ2,e2]\nδAC = (1 / 2π) Σj,e [|Ωj,e2|²/(4Δj,e2) − |Ωj,e1|²/(4Δj,e1)]</pre>",
        "<p>上述 Ω、Δ 在计算中都用 rad/s，报告的 Ωeff/(2π) 用 Hz 或 kHz；有效哈密顿量非对角元约定为 ℏΩeff/2。F′=0、1、2、3 的全部磁子态都进入计算，不同中间态先相干求和再取模。当前钟态和偏振下，仅 F′=1、2 有非零 Raman 乘积；F′=0、3 仍参与单光子光移和散射。两束光对两个基态的非主支路耦合也已纳入。</p>",
        _table(["方案（第一/第三光区）", "P1/P2", "P1 / mW", "P2 / mW", "Ωeff/(2π) / kHz", "差分光移 / kHz"], compare),
        "<p>相同拉比频率不代表相同共振条件。1:1 功率对照虽然达到同一 Ωeff，但有上表所示的差分光移，不能直接替代此前差分光移为零的优化配置。" + cancellation_note + "</p>",
        "<img src='power_curve.png' alt='实际两色总功率与有效拉比频率及差分光移的关系'>",
        "<h2>散射与近似范围</h2>",
        _table(["光区", "F=1,m=0 峰值散射率 / s⁻¹", "F=2,m=0 峰值散射率 / s⁻¹"],
            [[p["zone"], _number(p["clock_F1_m0_peak_scattering_rate_per_s"]), _number(p["clock_F2_m0_peak_scattering_rate_per_s"])] for p in powers]),
        "<p>这些是光斑中心、原子处于指定基态时的峰值速率，不是一次穿越的散射光子数。当前新增的是散射诊断，Raman 传播仍使用有效钟态双能级相干模型；此前的条纹和最优点不能被称为加入 Raman 自发散射后重新优化的结果。</p>",
        "<p>第一/第三光区最小允许跃迁失谐为 %s MHz，最大单光子 |Ω/Δ| 为 %s；中间光区为 %s。代码要求所有有光照的允许跃迁满足 |Δ|≥20Γ，且最大 |Ω/Δ|≤0.1；不满足时拒绝用该远失谐消去模型给出功率。</p>" %
        (_number(selected["min_abs_detuning_Hz"] / 1e6), _number(selected["max_single_photon_rabi_to_detuning_ratio"]), _number(data["zone_calibrations"][1]["max_single_photon_rabi_to_detuning_ratio"])),
        "<h2>运行与数据</h2>",
        "<pre>python -m rb87_beam.raman_power_report\npython -m rb87_beam --config beam_outputs/raman_power_calibration/config_power_driven.json --experiment phase --output beam_outputs/raman_power_replay</pre>",
        "<p>第一条命令读取当前已优化配置并生成本报告。第二条从所给配置的源模块运行，可能重新计算 MOT 和光泵浦；若使用已有 prepared 检查点，应同时指定 --source-state-h5 和 --no-optical-pumping，避免重复制备。配置中的 beam1_power_W 和 beam2_power_W 用 W，表格和 CSV 用 mW；修改实际功率会重新计算耦合和光移，而不是继续使用旧的 peak_rabi_Hz。</p>",
        "<p>导出的示例 Rabi 扫描范围为 0–2π，单次穿越示例总面积为 2π；没有直接继承旧的 6π 强驱动范围。较强驱动需要检查各实际功率是否仍满足远失谐条件，超出适用范围时程序会明确报错。</p>",
        "<p><a href='powers.csv'>三光区功率 CSV</a> · <a href='power_curve.csv'>功率扫描 CSV</a> · <a href='calibration.json'>完整计算记录</a> · <a href='config_power_driven.json'>实际功率驱动配置</a> · <a href='power_curve.png'>科学图</a></p>",
        _verification_html(data, output),
        "<h2>数据与公式来源</h2>",
        "<p><a href='https://steck.us/alkalidata/rubidium87numbers.pdf'>Steck：Rubidium 87 D Line Data</a>，约化偶极矩约定与D2矩阵元。这里采用 CG 约定的 dS = %s C·m；PyLCP归一化循环跃迁振幅乘以 dS/√2，避免与标准3j约化矩阵元 √2 dS 混用。</p>" % _number(STECK_D2_CG_REDUCED_CM),
        "<p><a href='https://jila.colorado.edu/~junye/yelabsOLD/pubs/scienceArticles/1990-1998/sArticle_1996_08_RbHyperfine.pdf'>Ye 等，Optics Letters 21, 1280 (1996)</a>，激发态超精细常数；此处使用共享 PyLCP 原子模型中的一致 A、B 常数计算能级，未混用不同来源的单独间隔。</p>",
        "<p><a href='https://arxiv.org/pdf/1605.05230'>Naber 等：Raman transitions between hyperfine clock states in a magnetic trap</a>，多路径相干求和与光移处理。其目标磁子态不同，本程序没有套用其具体 CG 系数。<a href='https://web.stanford.edu/group/kasevich/cgi-bin/wordpress/wp-content/uploads/2022/04/Y-WU_2021_PhD_Thesis.pdf'>Yunfan Wu 博士论文（2021），附录A</a>讨论钟态 Raman 的偏振选择。</p>",
        "<div class='scope'>以上结果是在声明的D2多能级、两色纯频率分量、弱场和理想自旋输运假设下的理论标定。它不是激光器功率实测值，也不替代实验中的光斑、偏振和光路损耗标定。</div>",
        "</body></html>"]
    (output / "report.html").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="beam_outputs/endpoint_comparison/joint_config.json")
    parser.add_argument("--output", default="beam_outputs/raman_power_calibration")
    parser.add_argument("--reference-fprime", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--detuning-hz", type=float, default=None)
    parser.add_argument("--power-ratio", default="cancel_light_shift", help="cancel_light_shift or P_high/P_low")
    args = parser.parse_args()
    ratio = args.power_ratio
    if ratio != "cancel_light_shift":
        try:
            ratio = float(ratio)
        except ValueError:
            parser.error("power-ratio must be cancel_light_shift or a positive number")
    data = generate_report(args.config, args.output, args.reference_fprime, ratio, args.detuning_hz)
    for row in data["powers"]:
        print("Zone %d: P_high=%.9f mW, P_low=%.9f mW, Omega/(2pi)=%.6f kHz" %
              (row["zone"], row["beam1_high_frequency_power_mW"],
               row["beam2_low_frequency_power_mW"], row["peak_rabi_kHz"]))
    print("Report:", (Path(args.output).resolve() / "report.html"))


if __name__ == "__main__":
    main()
