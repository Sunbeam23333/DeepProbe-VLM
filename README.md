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

```bash
git clone https://github.com/Sunbeam23333/DeepProbe-VLM.git
cd DeepProbe-VLM/experiment
# 先准备兼容服务器 GPU 的 PyTorch，再安装这些依赖。
python -m pip install -r requirements/train.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

从旧目录布局升级后，需要从 `experiment/` 重新执行 editable 安装。已有数据、缓存和运行目录不会自动迁移；保留原目录，使用明确路径，或自行安排存储迁移。不要因拉取代码而覆盖已有检查点。

## 当前状态

已实现：Ouro 原生循环、连接器对齐、全主干 SFT、普通模型对照、数据准备、参考评测与断点续训。

截至 **2026-11-17**，已验收的模型评测仍为 R1/R2 seed17：两者已完成两阶段训练，Video-MME 的声明 899 视频子集准确率为 **6.3404% / 18.4650%**，TempCompass 多选准确率为 **42.5316% / 50.4430%**。Video-MME 包含较高无效回答率，不代表官方全量成绩；协议、配对统计、局限与证据摘要见[结果记录](experiment/results/RESULTS.md)。

R4 alignment 已完成；SFT 保存了 600/718 步检查点，但当前本地缺少其中 5 个模型/优化器文件，恢复仍需从已验收归档还原并通过容量门槛。Smol seed29 alignment a2 的失败归档及完整 300 步检查点已独立验收；a3 经准入后以单节点八卡 low 作业提交，11 月 17 日北京时间 **14:20:58** 开始运行、**14:25:50 Failed**。八卡身份、BF16 前后向和 NCCL 检查已通过，但 preflight 进程运行 58.5 秒后收到 SIGTERM，未正常退出；尚无正式训练启动、原生恢复或新增步数证据，信号来源未确认。23:16 的回读确认当前 6 个课题 Job、3 个 Pod 均为终态。两者尚无完整模型评测成绩。额外训练种子、剩余消融及正式模型成本/延迟实验仍待完成，详见[当前状态](experiment/results/STATUS.md)。

CUDA QA a3 已通过八卡启动自检、真实 BF16/FP32 池化数值与单算子计时检查，并完成独立归档验收。这些结果覆盖单卡合成输入的池化算子；完整模型、分布式训练及 H20/B300 对比的速度结论仍待测量。[完整串行矩阵](experiment/docs/EXPERIMENT_SUITE.md)包含 185 个计划任务，尚未提交；单独的 Smol 恢复不等于启动整套实验。

训练恢复现在预检 optimizer/scheduler/RNG 等必需状态；推理新增显式启用的最后位置 logits 投影、Triton 融合空间池化和独立 profiler 诊断。默认训练及参考评测保持原路径。实现范围、已验收的 CUDA 证据与待补的模型测量见[优化记录](experiment/docs/OPTIMIZATION_STATUS.md)。选择性视觉刷新、token compaction、跨轮 KV 共享和 TTT 尚未实现；新增池化核不等于这些研究机制已经完成。

Git 只同步代码、测试和经过核对的结果摘要。数据集视频、模型权重、私人邮件、学生身份信息及原始集群快照不在仓库中。

![Current model overview](paper/figures/final/main_overview.png)

论文源码和图的源文件/成图均跟随 Git 同步；运行产物默认忽略。真实实验完成后，再将经过审查的汇总结果与图加入 `paper/`，不要把原始训练目录整体提交。
