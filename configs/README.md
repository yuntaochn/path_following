# baseline_config.json 配置说明

`baseline_config.json` 是所有实验参数的**唯一权威源**。  
所有 `scripts/exp_0X_*.py` 脚本通过 `get_config()` 读取它，不在脚本里硬编码数值。

> JSON 不支持注释，本文件就是配置的"注释层"。每次修改 JSON，建议同步更新这里。

---

## 顶层结构

```
baseline_config.json
├── simulation          仿真引擎参数（步长、积分方法等）
├── metrics             性能指标计算窗口和阈值
├── initial_state       所有实验的统一初始状态
├── model               船舶模型选择
├── los                 ILOS 导引律参数（基线默认值）
├── pid                 PID 控制器参数（三套预设）
├── actuator            双桨推进器物理参数
├── controller          控制器总装参数
├── paths               路径点定义（3 条论文路径）
├── disturbances        扰动场景定义（3 种）
├── shapers             航向整形器配置（3 种方法）
├── velocity_schedulers 速度调度器配置（2 种模式）
├── methods             实验方法定义（7 种，引用上面的模块）
├── experiments         5 个论文实验定义
├── plotting            matplotlib 全局样式
└── summary_fields      CSV 输出的指标列名
```

---

## simulation — 仿真引擎参数

| 字段 | 值 | 说明 |
|------|----|------|
| `dt` | 0.05 s | 仿真步长（20 Hz），RK4 精度足够 |
| `t_final` | 200.0 s | 单次仿真最长时间，防止死循环 |
| `integration_method` | `"rk4"` | 数值积分方法，`"euler"` 精度较低 |
| `goal_tolerance` | 3.0 m | 到达终点的判定半径（论文"3 m 容差圆"） |
| `stop_when_reached` | true | 进入容差圆后立即停止，节省计算时间 |

---

## metrics — 性能指标计算参数

这些字段控制 `src/baseline/metrics.py` 的计算窗口，**不影响仿真本身**。

| 字段 | 值 | 说明 |
|------|----|------|
| `analysis_start_time` | 15.0 s | 跳过仿真前 15 s 的启动瞬态，CTE-RMS 只统计此后数据 |
| `turn_pre_window` | 1.0 s | 转弯指标窗口：切换前 1 s |
| `turn_post_window` | 8.0 s | 转弯指标窗口：切换后 8 s（覆盖超调恢复段） |
| `cross_track_settling_band` | 0.5 m | CTE 稳定判定带宽（±0.5 m 内视为收敛） |
| `heading_settling_band` | 0.0873 rad | 航向稳定带宽，约等于 **5°**（= 5×π/180） |
| `tau_r_limit_config` | 9.0 N·m | 饱和时间统计用的偏航力矩阈值（= τ_r,max） |

---

## initial_state — 初始状态

所有实验共用，脚本从这里读，不硬编码。

| 字段 | 值 | 说明 |
|------|----|------|
| `eta0` | `[0, -10, 0]` | 初始位置 [x, y] = (0, −10 m)，艏向 ψ = 0 rad（正东） |
| `nu0` | `[0, 0, 0]` | 初始速度全零（从静止出发） |

> 为什么 y = −10？让船从路径起点稍靠下方出发，有短暂收敛过程，更贴近实际。

---

## model — 船舶模型

| 字段 | 值 | 说明 |
|------|----|------|
| `type` | `"cs2"` | 使用 CyberShip II 精确参数（对应 `USV3DOF` 类） |

CS2 物理参数固化在 `src/baseline/usv_model.py` 的 `CS2Params` 数据类中，无需在配置文件里重复。

---

## los — ILOS 导引律参数

这是**基线方法**（`los_pid`, `fixed_rate`, `first_order`）使用的默认前视距离。  
SHCS 系列方法通过 `methods.*.los_override` 将 `lookahead` 覆盖为 4.0 m。

| 字段 | 值 | 说明 |
|------|----|------|
| `type` | `"ilos"` | 使用积分 LOS（补偿稳态海流偏流） |
| `lookahead` | 10.0 m | 前视距离 Δ（基线保守配置，论文 Δ=10 m） |
| `k_i` | 0.05 | ILOS 积分增益，控制海流估计速度 |
| `sigma_limit` | 3.0 | 积分项上限 |σ_i| ≤ 3，防止过度积分 |
| `switch_radius` | 3.0 m | 航点切换触发距离（进入此半径则切换至下一航点） |

---

## pid — PID 控制器参数

定义了三套 PID 预设，由 `methods` 节的 `pid_heading` 字段选择。

### `pid.surge`（纵荡速度 PID）

控制实际速度 u 跟踪期望速度 u_d，输出纵荡推力 τ_u。

| 字段 | 值 | 说明 |
|------|----|------|
| `kp/ki/kd` | 20 / 2 / 1 | 比例/积分/微分增益 |
| `output_limit` | [−60, 60] N | 输出限幅 = τ_u,max（双桨最大纵荡推力） |
| `angle_wrap` | false | 速度误差不需要角度归一化 |

### `pid.heading`（标准航向 PID，默认）

控制实际艏向 ψ 跟踪参考 ψ_ref，输出偏航力矩 τ_r。

| 字段 | 值 | 说明 |
|------|----|------|
| `kp/ki/kd` | 30 / 3 / 8 | 比例/积分/微分增益 |
| `output_limit` | [−9, 9] N·m | 偏航力矩限幅 = τ_r,max = T_max × b |
| `angle_wrap` | true | 航向误差需 wrap_to_pi 归一化（防止 ±π 跳变） |

### `pid.heading_aw`（带抗积分卷绕航向 PID）

仅用于消融实验中的 `AW` 方法，其余参数与 `heading` 完全相同。

| 字段 | 值 | 说明 |
|------|----|------|
| `aw_gain` | 1.0 | 反算抗卷绕增益（AW 修正量 = aw_gain × 超限误差） |

---

## actuator — 双桨推进器参数

| 字段 | 值 | 说明 |
|------|----|------|
| `T_max` | 30.0 N | 单桨最大推力 |
| `b` | 0.30 m | 两桨横向间距 |
| `note` | — | 推导结果：τ_r,max = 9 N·m，τ_u,max = 60 N |

这两个值驱动论文式(3)(4)中的联合约束。速度调度器的 `T_max`/`b` 字段也应与此保持一致。

---

## controller — 控制器总装参数

| 字段 | 值 | 说明 |
|------|----|------|
| `u_d` | 1.5 m/s | 额定巡航速度（speed scheduler 的 u_nominal） |
| `tau_r_filter_tau` | 0.06 s | 偏航力矩输出一阶低通滤波时间常数（防推进器抖振） |
| `tau_r_rate_limit` | 220.0 N·m/s | 偏航力矩变化率上限（与低通滤波配合使用） |

---

## paths — 路径定义

论文使用的 3 条路径，新路径直接在这里添加。

| 名称 | 类型 | 说明 |
|------|------|------|
| `l_shape` | `points` | L 形，单次 90° 转弯：(0,0)→(50,0)→(50,50) |
| `double_l` | `points` | 双 L 形，两次 90° 转弯，用于验证连续折线 |
| `s_curve` | `s_curve` | 正弦 S 形，验证方法不仅适合折线路径 |

`s_curve` 的参数含义：
- `length` = 120 m：路径总长度
- `amplitude` = 15 m：侧向最大摆幅
- `num_points` = 20：路径点数（越多越平滑）
- `periods` = 1.5：1.5 个完整正弦周期

---

## disturbances — 扰动场景

> **省略字段 = 零**：未写出的噪声/偏置字段在加载时自动填充零向量（`config.py` 的默认值机制）。

| 名称 | 内容 | 使用场景 |
|------|------|---------|
| `calm` | 无扰动（全零，JSON 为空 `{}`） | 泛化实验（Sec 4.3）无扰动组 |
| `steady_current` | 定常海流 [0.35, 0.18] m/s | 典型场景（Sec 4.2）、消融、敏感性分析 |
| `current` | 定常海流 + 随机力噪声 + 传感器噪声 | 蒙特卡洛（Sec 4.4）、泛化实验含噪组 |

`current` 的噪声参数含义：
- `force_noise_std: [1.2, 0, 0.2]`：纵荡力噪声 1.2 N，偏航力矩噪声 0.2 N·m（模拟波浪）
- `eta_noise_std: [0.1, 0.1, 0.0175]`：位置噪声 ±0.1 m，航向噪声 ≈ **1°**（= 0.0175 rad）
- `nu_noise_std: [0.05, 0.02, 0.007]`：速度测量噪声

---

## shapers — 航向整形器

定义论文 Sec 3.2 中对比的三种整形方法：

| 名称 | 方法 | 关键参数 |
|------|------|---------|
| `dynamic` | 动态整形（本文方法） | `M33`=2.76, `Nr`=−1.9, `Nrr`=−0.75, `tau_r_max`=9, `r_nominal`=1.5 |
| `fixed_rate` | 固定速率限幅（基线 FR） | `r_fixed`=0.3 rad/s（离线整定） |
| `first_order` | 一阶参考模型（基线 FO） | `T_filter`=2.0 s（滤波时间常数） |

`dynamic` 参数来自 CS2 偏航动力学：
- `M33 = Iz - N_dr = 1.76 - (−1.0) = 2.76 kg·m²`（有效偏航惯量）
- `Nr = −1.9 N·m·s/rad`，`Nrr = −0.75 N·m·s²/rad²`（偏航阻尼，负值）
- `r_nominal = 1.5 rad/s`：整形速率上限（论文敏感性分析默认值）

---

## velocity_schedulers — 速度调度器

| 名称 | 模式 | 说明 |
|------|------|------|
| `shcs` | `coupled` | 联合约束感知模式（使用 preview 接口） |
| `shcs_simple` | `simple` | 仅残差驱动降速（不需要推进器模型） |

两种模式共用参数：

| 字段 | 值 | 说明 |
|------|----|------|
| `u_nominal` | 1.5 m/s | 额定速度（与 `controller.u_d` 一致） |
| `u_min` | 0.3 m/s | 最小允许速度（防止完全停车） |
| `lambda_schedule` | 0.6 | 降速强度系数（论文参数 λ，0~1，越大转弯减速越多） |
| `e_s_max` | 1.5708 rad | 整形残差最大值 = **π/2**（90°），用于归一化 |
| `e_s_deadband` | 0.0873 rad | 残差死区 = **5°**，小于此值不降速 |
| `tau_smooth` | 0.5 s | 速度命令平滑时间常数（防突变） |
| `u_d_rate_limit` | 1.0 m/s² | 速度命令变化率上限 |

`shcs`（coupled 模式）额外参数：

| 字段 | 值 | 说明 |
|------|----|------|
| `T_max` | 30.0 N | 单桨最大推力（与 `actuator` 一致） |
| `b` | 0.3 m | 推进器间距（与 `actuator` 一致） |
| `tau_u_limit` | 60.0 N | 最大纵荡推力 = 2×T_max |
| `tau_r_deadband` | 9.0 N·m | 偏航力矩死区（低于此值认为无偏航需求） |

---

## methods — 实验方法

每个方法通过**引用**上面定义的模块来组合，而不是重复写参数。

| 方法名 | 标签 | 整形器 | 速度调度 | 前视距离 |
|--------|------|--------|---------|---------|
| `los_pid` | ILOS | 无 | 无 | 默认 10 m |
| `fixed_rate` | ILOS+FR | `fixed_rate` | 无 | 默认 10 m |
| `first_order` | ILOS+FO | `first_order` | 无 | 默认 10 m |
| `dynamic_shaper` | DS | `dynamic` | 无 | **覆盖 4 m** |
| `anti_windup` | AW | 无 | 无 | 默认 10 m（使用 `heading_aw` PID） |
| `shcs_simple` | SHCS-Simple | `dynamic` | `shcs_simple` | **覆盖 4 m** |
| `shcs` | SHCS | `dynamic` | `shcs` | **覆盖 4 m** |

`los_override.lookahead: 4.0` 只覆盖使用该方法时的前视距离，不影响其他方法。  
这是"短前视 + 整形协同"设计的关键：SHCS 获得更好的几何精度（短前视），整形层防止因此产生的偏航饱和。

---

## experiments — 5 个论文实验

每个实验说明要用哪些方法、路径和扰动，并指定输出目录。

| 实验名 | 输出目录 | 论文章节 | 关键设计 |
|--------|---------|---------|---------|
| `compare_methods` | `01_typical_l_shape/` | Sec 4.2 | 4 种方法 × L 形 × 定常海流 |
| `path_generalization` | `02_path_generalization/` | Sec 4.3 | 3 种方法 × 3 路径 × 3 扰动 |
| `monte_carlo` | `03_monte_carlo/` | Sec 4.4 | 2 种方法 × 20 随机种子（配对） |
| `sensitivity` | `04_parameter_sensitivity/` | Sec 4.5 | SHCS × λ 和 r_nominal 扫描 |
| `ablation` | `05_ablation/` | Sec 4.6 | 5 级消融（基线→完整 SHCS） |

`sensitivity.scan` 字段定义了参数扫描的取值列表：
- `lambda_schedule: [0.1, ..., 1.0]`：10 个等间距值
- `r_nominal: [0.3, 0.5, ..., 2.5]`：8 个值，覆盖"过保守→合适→过宽松"区间

---

## plotting — 绘图样式

控制所有实验图的全局 matplotlib 样式，修改这里可统一改变所有图的外观。

| 字段 | 值 | 说明 |
|------|----|------|
| `matplotlib_cache_dir` | `/tmp/...` | 防止字体缓存写入项目目录 |
| `figsize` | [11.0, 8.0] | 默认图幅（英寸） |
| `dpi` | 180 | 输出分辨率（300 DPI 以上适合论文） |
| `trajectory_linewidth` | 2.0 | 轨迹线宽 |
| `grid_alpha` | 0.3 | 网格透明度 |

---

## summary_fields — 输出指标列表

控制 `experiment_utils.save_summaries_csv()` 写入 CSV 的列名顺序。  
这些字段名对应 `src/baseline/metrics.py` 中 `summarize_tracking_log()` 返回的字典键。

关键字段说明：

| 字段 | 说明 |
|------|------|
| `cross_track_rms` | CTE-RMS（论文主要指标） |
| `heading_error_rms` | 航向参考跟踪 RMS（跟踪 ψ_ref，非 ψ_d） |
| `control_energy_tau_r_cmd` | 偏航控制能耗 ∫τ_r² dt |
| `turn_cte_rms` | 转弯窗口内的 CTE-RMS（反映转弯超调） |
| `turn_sat_ratio_tau_r_cmd` | 转弯窗口内的饱和时间占比 |
| `speed_reduction_max_pct` | 最大降速比（SHCS 速度调度效果） |
| `reached_goal` | 是否进入 3 m 终点容差圆（严格到达判定） |
| `path_completed` | 是否完成路径（宽松判定，包含越过终点截面） |
