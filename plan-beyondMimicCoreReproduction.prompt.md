## Plan: BeyondMimic核心结果复现（单机单卡）

目标是在单机单卡条件下，复现BeyondMimic论文的核心实验链路与关键结果：数据预处理、单动作到多动作训练、定量指标复现、可视化对齐，并形成可重复执行的实验记录。优先复用仓库现有实现，补齐评测与实验管理缺口，最后再考虑sim2real外延。

**Steps**
1. Phase 0 - 基线冻结与环境验收
1.1 固定软件栈与随机种子：Isaac Lab/Isaac Sim/Python/WandB版本、GPU驱动、仓库commit与依赖锁定。
1.2 验证最小可运行闭环：CSV到NPZ、NPZ回放、单动作训练、训练后play回放与ONNX导出。
1.3 建立实验目录与命名规范（motion名、run名、WandB project/tag），避免结果污染。

2. Phase 1 - 论文实验映射与缺口定义（depends on 1）
2.1 从论文抽取必须复现的核心结果项：主任务、对比设置、评价指标、可视化类型。
2.2 将论文术语映射到现有代码模块：奖励项、观测项、终止条件、训练超参、任务变体。
2.3 输出差距清单：仓库已有项、需补实现项、可近似替代项、暂不纳入项（明确边界）。

3. Phase 2 - 数据与动作集构建（depends on 1, parallel with 2部分）
3.1 选定核心动作子集（先中等难度+高动态动作混合），定义训练/验证动作划分。
3.2 标准化CSV转NPZ参数（输入fps、输出fps、frame range、是否headless），保证可重复。
3.3 建立动作清单元数据（motion长度、节奏类别、难度标签），用于后续采样与统计。

4. Phase 3 - 训练协议复现（depends on 2 and 3）
4.1 确定单机单卡资源预算：num_envs、max_iterations、batch策略与显存监控阈值。
4.2 先完成单动作基线，再推进多动作训练策略（按难度分阶段或按类别混合采样）。
4.3 统一训练日志协议：记录奖励分量、跟踪误差、终止原因分布、吞吐与稳定性指标。
4.4 复现实验矩阵：至少包含主配置+关键消融（如低频控制、无状态估计变体）。

5. Phase 4 - 评测体系补齐（depends on 4）
5.1 实现定量评测脚本：轨迹跟踪误差、姿态误差、速度误差、早停率、成功率等。
5.2 定义论文对齐统计口径：均值、方差、置信区间、跨seed聚合方式。
5.3 生成可视化产物：训练曲线、动作回放视频、关键帧对齐图（策略 vs 参考动作）。

6. Phase 5 - 结果对齐与复现实验报告（depends on 5）
6.1 对比论文核心结果：逐项标注“已对齐/部分对齐/未对齐”与偏差解释。
6.2 做误差归因：数据质量、控制频率、奖励权重、随机化强度、硬件算力限制。
6.3 形成最终复现实验报告：配置清单、命令清单、图表与结论、复现实验脚本入口。

7. Phase 6 - 可选外延（excluded from core reproduction）
7.1 sim2real控制器联动与实机验证（需外部仓库 motion_tracking_controller 与硬件条件）。
7.2 对称性训练、扩散控制等论文扩展能力的深度复现。

**Relevant files**
- /home/imagineerfeng/whole_body_tracking/README.md - 现有运行链路与命令入口，作为复现实验主流程模板。
- /home/imagineerfeng/whole_body_tracking/scripts/csv_to_npz.py - CSV到NPZ预处理与动作资产生成。
- /home/imagineerfeng/whole_body_tracking/scripts/replay_npz.py - 动作资产回放验证。
- /home/imagineerfeng/whole_body_tracking/scripts/upload_npz.py - WandB动作资产上传备用链路。
- /home/imagineerfeng/whole_body_tracking/scripts/rsl_rl/train.py - 主训练入口与资源参数控制。
- /home/imagineerfeng/whole_body_tracking/scripts/rsl_rl/play.py - 评估回放与模型导出入口。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py - 环境级配置（并行环境数、频率、episode）。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py - G1任务与动作/观测细节配置。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py - 奖励项定义与权重复现关键点。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py - 观测定义与状态估计/特权信息切分。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py - 失败判定与早停条件。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py - 动作命令加载与采样逻辑。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py - 训练运行器与导出/WandB联动。
- /home/imagineerfeng/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/exporter.py - ONNX导出及元数据封装。
- /home/imagineerfeng/whole_body_tracking/data/LAFAN1/dataset_infos.json - 数据集信息源与动作索引。

**Verification**
1. 环境验证：完成一次最小闭环并保留日志（CSV转NPZ成功、回放正常、训练启动并收敛趋势正常、play可加载checkpoint）。
2. 重复性验证：同一配置3个seed复现实验，关键指标方差在可接受区间内。
3. 论文对齐验证：至少1组主结果与1组消融结果在统计口径上可对比，并给出偏差分析。
4. 产物验证：生成并归档训练曲线、视频、指标表格与最终复现报告。

**Decisions**
- 目标定义：优先复现论文核心结果，不以实机部署作为本阶段成功标准。
- 资源约束：按单机单卡设计实验矩阵，优先保证结果可重复而非覆盖全部动作。
- 范围边界：核心阶段不强制实现扩散控制与完整sim2real流程，但需在报告中说明缺失影响。

**Further Considerations**
1. 动作子集策略建议：A 先静态/低动态再逐步加高动态；B 直接混合训练但增加训练时长。推荐A以降低单卡不稳定风险。
2. 结果对齐标准建议：A 严格数值接近论文；B 先保证趋势与相对排序一致。推荐B作为第一轮里程碑。
3. 评测优先级建议：A 先实现误差与成功率；B 同时实现完整时序稳定性指标。推荐A先落地，再扩展B。