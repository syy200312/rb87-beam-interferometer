# ⁸⁷Rb 连续冷原子束拉曼干涉仪仿真

Python 仿真覆盖 2D⁺ MOT 多能级冷却、再泵浦与频率调制、向 |F=1, mF=0⟩ 的光泵浦态制备、空间三脉冲拉曼干涉及双通道荧光探测。可运行单区拉比振荡、相位条纹、变啁啾条纹，以及入射通量与条纹拟合精度扫描。软件采用单束模型，提供离线角速度读出和探测白噪声预报。随附的 HDF5/NPZ 是仿真输入样例。

## 安装和首次运行

在 Windows 使用 64 位 CPython 3.11。从本仓库解压后的根目录或克隆目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_demo.py
```

macOS/Linux 可用相同版本的 Python 建立环境：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_demo.py
```

以下示例的 `python` 指这个虚拟环境中的解释器。在 VS Code 用 **Python: Select Interpreter** 选择 `.venv`，在“运行和调试”选择 **Rb87 00** 按 F5。安装依赖需要联网。

## 第一个结果

`run_demo.py` 加载随附的 256 个制备后原子的坐标、速度与密度矩阵，关闭随机探测噪声，在同一批原子上运行以下三个扫描。每个目录都有 CSV 原始数值和 JSON 配置或摘要。可以用 `python run_demo.py --output outputs/my_scan` 改变输出目录。

| 文件 | 横轴与读法 |
|---|---|
| `outputs/demo/phase/overview.png` | 第三光区附加相位（rad），荧光归一化读数的正弦拟合对比度约 48.79% |
| `outputs/demo/rabi/rabi.png` | 单区拉曼耦合强度，双光束总功率约 0.620 mW 时出现首峰 |
| `outputs/demo/chirp/chirp.png` | 双光子频差啁啾率（Hz/s），固定第三脉冲附加相位；扫描极值可见度约 57.44% |

## 从原子源开始

```sh
python -m rb87_beam --config beam_configs/optical_pumping_optimized.json --atoms 8 --no-noise --output outputs/fresh
python -m rb87_beam --config beam_configs/optical_pumping_optimized.json --source-state-h5 beam_outputs/pylcp_multilevel_32/source_ensemble.h5 --compare-preparation --no-noise --output outputs/prep_comparison
python -m rb87_beam --help
```

第一条命令重新计算多能级 MOT、光泵浦、拉曼干涉和荧光，而第二条从 MOT 出口比较不同的态制备方案。8 个原子适合检查程序能否运行，做物理比较时应增大独立源样本数，检查数值步长和误差。

## 重放、参数扫描和结果

要在相同有效通量下比较双光子相位、反射镜位移、纵向磁场和光强噪声，在 VS Code 选择 **Rb87 10**，或运行 `python -m rb87_beam.raman_noise_study`。默认输出到 `beam_outputs/raman_noise/`。噪声幅度与三区相关性均可编辑。计算约定和输出字段见[三脉冲噪声使用说明](README_Raman噪声.md)。

| 想改变的设置 | 入口或字段 |
|---|---|
| 原子数和随机种子 | `--atoms`、`--seed`，用于新生成的源。HDF5 重放保留其原子集合 |
| 拉曼功率 | 在复制出的 `config_power_driven.json` 中修改 `raman.beam1_power_W`、`raman.beam2_power_W`（W）。保留功率模式并将独立峰值覆盖设为 0 |
| 啁啾率 | `--experiment chirp --chirp-center-hz-s 0 --chirp-half-range-hz-s 2500000 --chirp-points 121`。这些参数需附加到使用所选输入和配置的主程序命令 |
| 均匀改变入射通量 | `rb87_beam.flux_visibility_study`。固定源位置、速度和内态，扫描荧光计数和统计拟合精度 |

```sh
python -m rb87_beam.flux_visibility_study --fluxes 1e4 1e5 1e6 1e8 1e10 --gate-ms 10 --repeats 100 --output outputs/flux
python -m rb87_beam.obe_80_study --suite nominal --atoms 2 --workers 1 --output outputs/obe_demo
python -m tmp.evaluate_gyro_sensitivity --output outputs/gyro.json
python -m unittest discover -s tests -v
```

用浏览器打开 `outputs/flux/flux_visibility.html` 可切换入射通量、光泵浦工作点和探测孔径。OBE 命令以两个源原子试运行独立 24 态光学布洛赫方程，具体统计结论需要增加样本并加密时间和空间展开。灵敏度示例使用固定工作相位、10 ms 积分门、独立白噪声模型。输入通量是配置中的归一化量。

## 进一步计算

`beam_configs/` 是可编辑 JSON，`rb87_beam/` 是仿真与分析源码，`tests/` 是回归测试。`beam_outputs/` 包含运行上述例程和 VS Code 启动项所需的仿真样例。重新计算的结果宜用 `--output outputs/名称` 分别保存。

```sh
python -m rb87_beam.raman_power_report --output outputs/power
python -m rb87_beam.optimization --help
python -m rb87_beam.joint_optimization --help
python -m rb87_beam.endpoint_comparison --help
python -m rb87_beam.low_power_study --help
```

对制备区和三脉冲进行联合搜索时，先用 `joint_optimization` 生成训练结果，再把该目录传给 `endpoint_comparison --study`。保存的单原子中间结果可以用于断点续算。扫描长实验请限制同时运行的进程以及 BLAS 的线程数。已制备的输入文件 `prepared_*.h5` 重放时请传 `--no-optical-pumping`，以免原子再次通过制备区。输出中 `fit.contrast` 是无量纲的条纹对比度，`fit.phase_rad` 是可带负号的拟合相位。

## 许可与来源

新编写的束流代码采用 [MIT License](LICENSE)。PyLCP 的原始声明和传统代码背景见 [第三方说明](THIRD_PARTY.md)。
