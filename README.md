# DeepProbe-VLM

原生共享权重循环模型的视频 SFT 研究。实验与论文分别同步，代码不混入论文目录，训练数据与权重不进入 Git。

```text
DeepProbe-VLM/
├── experiment/                 # 实验：安装、数据、训练、评测与分析
│   ├── README.md               # 可执行的上机步骤
│   ├── docs/REQUIREMENTS.md     # 实验要求、运行顺序、验收与交付清单
│   ├── configs/                # 固定模型/数据版本与训练配置
│   ├── src/                    # 原生循环 VLM 实现
│   ├── scripts/                # 数据、启动、自检与统计工具
│   ├── tests/                  # CPU 契约测试
│   ├── requirements/
│   └── docker/
└── paper/                      # 论文：源文件、PDF、图与绘图源代码
    ├── README.md               # 文稿状态与编译说明
    ├── experiment_protocol.tex
    ├── experiment_protocol.pdf
    ├── figures/source/         # 可编辑 XeLaTeX/TikZ 图
    ├── figures/final/          # PDF / PNG / SVG 成图
    ├── figures/candidates/     # 六份布局候选与总览
    └── docs/FIGURE_AUDIT.md
```

## 从哪里开始

- **跑实验**：先读 [实验要求清单](experiment/docs/REQUIREMENTS.md)，再按 [上机指南](experiment/README.md) 执行。全部实验命令以 `experiment/` 为工作目录。
- **看论文和图**：进入 [paper/](paper/README.md)。当前同步的是五页实验协议及三张方法/流程图，不是已有结果的完整论文。
- **核对可信边界**：[已验证事项](experiment/docs/VALIDATION.md)、[模型契约](experiment/docs/MODEL_CONTRACT.md)、[实验矩阵](experiment/docs/EXPERIMENTS.md)。
- **看真实结果与进度**：[已验收结果](experiment/results/RESULTS.md)、[当前实验状态](experiment/results/STATUS.md)。
- **下载可复核的结果**：[2026-11-19 完整结果包](experiment/results/releases/2026-11-19/README.md)、[ModelScope 数据集](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM)。包含 8,554 条逐题记录、配对统计和 400 条 CUDA 算子计时；SHA 清单区分原始证据与脱敏副本。

```bash
git clone https://github.com/Sunbeam23333/DeepProbe-VLM.git
cd DeepProbe-VLM/experiment
# 先准备兼容服务器 GPU 的 PyTorch，再安装这些依赖。
python -m pip install -r requirements/train.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

从旧目录布局升级后，需要从 `experiment/` 重新执行 editable 安装。已有数据、缓存和运行目录不会自动迁移；保留原目录，使用明确路径，或自行安排存储迁移。不要因拉取代码而覆盖已有检查点。

## 迁移与备份

完整代码、数据、检查点及关键日志正在迁移到 [ModelScope 模型仓库](https://modelscope.ai/models/JasonJin/DeepProbe-VLM/files)。首批数据计划为 **226.68 GB**，已验收的本地代码与日志快照包含 **17,090 个文件**。下载、校验和换机恢复方法见[迁移说明](experiment/docs/MIGRATION.md)。数据传输尚未全部完成。

## 当前状态

**2026-11-21北京时间23:30最新核验**：a12 已完成 R4 两项评测和五种全模型延迟测量，但在无缓存输出 token 一致性检查失败后停止。冻结结果的 CPU 复核确认：无缓存、Triton 池化、最后 token 投影与 Triton 组合均未通过；最后 token 投影单独通过。失败配置不报告加速收益，权重与日志保留。数据迁移独立继续。见[一致性检查记录](experiment/results/R4_LATENCY_PARITY_20261121.json)。

截至 2026-11-21 北京时间 22:43，已验收的科学结果为 **7 个训练阶段、3 个完整两阶段模型、6 项评测**。R4 seed17 新完成的 Video-MME 899 视频子集准确率为 **20.3930%**，TempCompass 多选准确率为 **55.5063%**；见[本次评测记录](experiment/results/R4_EVALUATION_20261121.json)。该快照中全模型延迟任务仍在运行。R1/R2 seed17 的 Video-MME 声明 899 视频子集准确率为 **6.3404% / 18.4650%**，TempCompass 多选准确率为 **42.5316% / 50.4430%**。Video-MME 有较高无效回答率，不代表官方全量成绩；协议、配对统计及局限见[结果记录](experiment/results/RESULTS.md)。

Smol alignment a6 已完成 **718/718 步**，其独立归档恢复和套件历史导入均已验收，无需重复对齐训练；原 Job 在归档阶段 Failed 的历史保留。R4 SFT 已从 checkpoint600 完成最后 **118 步**，训练、原生检查点导出及完整阶段归档均已验收。完整计划保留 **18 个模型、34 个训练阶段、36 项评测、15 项全模型延迟、12 项比较**；剩余为 **27 个全新训练阶段、32 项新评测及上述延迟与比较**。

本轮 CUDA 门槛的合成池化算子测量中，BF16 的 reference → Triton 中位时间为 **0.249408 → 0.096992 ms（2.5714×）**，FP32 为 **0.242176 → 0.097776 ms（2.4768×）**，数值检查均通过。输入为 [8,729,1152]、G4，每后端 25 次预热、100 次计时、交替执行；计时包含输出分配。这些是单设备池化算子结果，尚不能说明全模型推理或训练加速。11 月 19 日原始结果包与旧微基准保持不变。

a6 终态元数据已归档验收：**458对象 / 1,946,568 B**。北京时间 **09:23:36** 仅释放旧 **48 GB预留额度**；其他42项预留和旧目录保留，没有删除48 GB文件。**09:24:53** 的项目 `/home` 实占为 **70.172 GB**，含预留及安全余量投影为 **72.443 GB / 150 GB**；这是写入退役收据前的历史快照，不是当前实时容量或新准入。OSS 不设容量上限。本地 checkpoint 仍须在完整归档校验及全部消费者完成后清理。

原有 Video-MME 与 TempCompass **F8 缓存已有 READY 验收**。**09:02:04** 的快照中，Video-MME899 F4 有 **677条元数据、2,708张 PNG，3,386对象 / 1,072,916,095 B**，尚无 READY；其余三个 F4/F16 形状尚无对象。缓存记录数不代表已验收视频或完成评测，当前进程是否存活不能由这些历史记录推出。

已实现原生 Ouro 循环、连接器对齐、全主干 SFT、普通模型对照、训练恢复及显式启用的最后位置 logits 投影、Triton 空间池化和独立 profiler。默认训练与参考评测沿用原路径。选择性视觉刷新、token compaction、跨轮 KV 共享和 TTT 尚未实现；范围和后续测量见[优化记录](experiment/docs/OPTIMIZATION_STATUS.md)。Git 同步经审查的代码与结果，数据集媒体、模型权重、身份信息及原始集群快照不进入仓库。

![Current model overview](paper/figures/final/main_overview.png)

论文源码和图的源文件/成图均跟随 Git 同步；运行产物默认忽略。真实实验完成后，再将经过审查的汇总结果与图加入 `paper/`，不要把原始训练目录整体提交。
