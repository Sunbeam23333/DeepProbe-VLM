# 实验要求与验收清单

更新：2026-10-08。**所有命令从 `DeepProbe-VLM/experiment/` 执行。**

当前交付是**稠密、原生共享权重循环的视频 SFT 基线**。选择性视觉更新、问题条件化视觉更新、跨轮 KV 共享、TTT、自定义 CUDA/PTX 尚未实现。下面的清单不代表这些创新实验已经完成。

## 1. 要验证什么

第一阶段回答三件事：原生循环模型能否被稳定适配到视频 QA；增加循环深度是否带来收益；收益对应多少实际计算成本。首先建立可信对照，不能只展示一个训练 loss 或一张速度图。

| 阶段 | 必跑内容 | 可以得出的结论 |
|---|---|---|
| P0 连通性 | 依赖/硬件自检 → 单卡 20-step pilot → 八卡 20-step pilot | 当前环境、数据路径、保存和通信是否工作 |
| P1 最小质量实验 | 对齐 → SFT；Ouro R=1/2/4；SmolLM2 外部参考 | 固定数据/步数下的质量与成本差异 |
| P2 消融 | 视觉编码器冻结/更新、帧数、空间 token 数、对齐阶段、cache/no-cache | 哪个设计影响质量或成本 |
| P3 论文结果 | 三 seed、完整覆盖率、真实硬件重复测量、误差与失败案例分析 | 在明确协议下可复核的研究证据 |

R=1/2/4 的相同步数比较**不是等计算量比较**。若声称更高效率，需补等预算或质量—成本曲线；SmolLM2 与 Ouro 的预训练、参数量和词表不同，只能作为外部参考。

## 2. 上机前准备与资源记录

- Linux、已能使用目标 GPU 的 PyTorch/CUDA 环境；Python 建议使用本项目验证过的 3.12 系列。
- 安装 `requirements/train.txt`，然后 `python -m pip install --no-deps -e .`。保留服务器适配的 GPU Torch，不用本地 CPU wheel 替换。
- 记录实际 GPU 型号/数量/显存、驱动、CUDA/PyTorch、CPU/RAM、共享内存、可用磁盘、数据/模型/输出所在盘。硬件型号标签不代表已通过兼容测试。
- 准备模型仓库与数据仓库的合法访问及足够缓存空间；模型、tokenizer、数据 revision 使用配置中的固定 SHA。
- 磁盘预算同时考虑：压缩包 + 解压视频 + 可选 PNG 帧缓存 + 模型缓存 + 中途检查点 + final。PNG 可以比原视频大很多；下载字节预算不覆盖全部解压/缓存成本。
- 当前为**复制式 DDP、FP32 参数及 Adam 状态、BF16 autocast**；不支持 ZeRO/FSDP。不得只按“参数量 × 2 bytes”估计显存，先 pilot 实测。

硬件检查具体命令见 [HARDWARE.md](HARDWARE.md)，安装与训练入口见 [../README.md](../README.md)。不在这里承诺 H20/B300 的显存门槛、小时数或速度倍数。

## 3. 数据要求

| 用途 | 数据 | 硬性约束 |
|---|---|---|
| 训练与开发 | LLaVA-Video-178K 的明确子集 | 按原始视频 ID 划分；同源片段和不同问题不跨 split |
| 最终测试 | Video-MME | 固定无字幕/无音频协议；报告 short/medium/long 与实际覆盖率 |
| 任务泛化 | MVBench | 明确任务清单、合法媒体、时间边界和缺失项；子集不冒充完整基准 |
| 可选补充 | HERBench | 单独注明版本、许可和媒体完整性；不把证据标签塞进提示词 |

下载、转换与来源映射以 [DATASETS.md](DATASETS.md) 为准。训练和验证候选要排除所有将用于最终测试的来源。验证集仅用于选模/调参；测试 gold answer 不进入模型提示，测试集不得用于 prompt 选择或早停。

数据就绪必须满足：

- [ ] 下载计划记录 revision、文件列表、字节预算和校验信息。
- [ ] manifest 可验证，媒体存在；重复/交叉来源检查通过。
- [ ] 记录 QA 数、独立来源视频数、帧数、长度分布、划分 seed 与 manifest SHA。
- [ ] 文本预算检查通过；任何筛选/截断政策都显式记录，不能偷偷删除长答案。
- [ ] 记录遗漏样本及原因；测试媒体缺失须中止或明确形成子集，不静默跳过。

### 推荐的统一帧缓存布局

默认配置 F=8。可以先缓存待划分的**训练候选集**，再按来源划分缓存 manifest，使训练/验证共用一个媒体根目录。此时只做确定性解码，不做任何训练；split 工具仍须排除最终测试来源。

```bash
python scripts/data/cache_frames.py \
  --input data/manifests/llava-pilot.jsonl --media-root data/media/llava \
  --output-dir data/frame_cache/llava-pilot-f8 --num-frames 8 --max-gb 100

python scripts/data/split.py \
  --input data/frame_cache/llava-pilot-f8/manifest.jsonl \
  --media-root data/frame_cache/llava-pilot-f8 \
  --output-dir data/manifests/llava-cached-split \
  --validation-fraction 0.05 --seed 42 \
  --exclude-manifest data/manifests/videomme-test.jsonl

# 每个最终会用到的 MVBench/HERBench manifest 再追加一个 --exclude-manifest。
python scripts/data/validate.py \
  data/manifests/llava-cached-split/train.jsonl \
  data/manifests/llava-cached-split/validation.jsonl \
  --media-root data/frame_cache/llava-pilot-f8
```

`100 GB` 是示例上限，不是预计占用或下载授权；运行前按自己的空间调整。训练参数改为对应的两个 cached-split manifest 与同一个 `--media-root`。帧数消融分别从原视频生成 F=4/8/16 缓存，不把 F16 缓存再次抽成 F8 当作相同的原始采样协议。

## 4. 先跑 P0，不直接铺开矩阵

`ouro_pilot.json` 固定 20 个**优化步骤**，accumulation=1；完整配方 accumulation=16。两者的实际 batch 与用途不同。

首次 pilot 从随机连接器开始，只验证连通性，不是效果对照。先单卡、再八卡，输出目录分开。正式质量训练要完成连接器对齐后再 SFT；`ouro_alignment.json` 是一轮对齐，**并不是 20-step 的短对齐配置**。

首轮建议显式准备一个小型、来源隔离的 pilot 子集，例如训练 128–256 条 QA、验证 16–32 条 QA。这是可调整的试跑预算，不是基准标准；必须记录实际 QA/来源数，且不能因此把同源视频放到两边。当前脚本不自动为你选择这个规模。

**注意：20 steps 不限制验证集大小。** 当前 pilot 在第 10、20 步及训练结束会评估验证集，验证过大会使小训练变成很长的作业。不要直接把大规模完整验证集当首轮 pilot 的验证输入。

通过标准：有限 loss、没有 NaN/OOM/解码异常；能够保存和重载；目标参数组获得有限梯度；恢复训练配置不变；NCCL/多卡通信正常。loss 不要求每一步单调下降。之后再做小样本过拟合和验证精度检查。CPU 小模型测试通过不能替代真实 GPU/checkpoint 的这些检查。

## 5. 正式训练前冻结任务配置

| 项目 | 当前默认值 / 要填写的内容 |
|---|---|
| 模型 | Ouro-1.4B + SigLIP；普通模型为 SmolLM2-1.7B-Instruct；具体 SHA 在 configs/train |
| 视觉输入 | F=8、G=4，即 128 视觉 token；文本预算 512 |
| 阶段 A | alignment，只更新池化/连接器，默认一轮 |
| 阶段 B | SFT，更新全部有效语言主干、视觉编码器和连接器；无用辅助 head 不训练 |
| 完整训练 batch | 8 GPU × microbatch 1 × accumulation 16 = global batch 128 |
| 学习率 | alignment 1e-3；SFT 1e-5；其他优化器/调度设置以 JSON 为准 |
| 训练预算 | 当前 max_steps=-1、epochs=1；须填子集大小和推导出的步数，不能只写“一轮” |
| seed | 先 17 跑最小矩阵；论文阶段补 29、43 |
| 选模规则 | 预先规定使用哪个验证指标/检查点；不能根据测试成绩选模型 |

当前配方不会自动加载“最佳验证检查点”，默认保存并使用训练末尾的 `final/`。若改成验证集选模，必须显式实现/记录规则，而不是训练结束后根据测试结果挑选。

第一次质量对比先跑 Ouro R=1、2、4 和普通模型参考；不是一次提交所有生成的配置。每个 R 使用匹配的对齐和 SFT，且初始化来自**同 R 的 alignment 输出**。当前加载器允许 stage initialization 时改变 R，所以此约束需要实验者遵守。

现有配置生成器可分别生成两个阶段：

```bash
python scripts/make_ablation_configs.py \
  --base configs/train/ouro_alignment.json --output-dir runs/configs/alignment
python scripts/make_ablation_configs.py \
  --base configs/train/ouro_sft.json --output-dir runs/configs/sft
```

比如 `alignment/loops_2_seed17.json` 对应 `sft/loops_2_seed17.json`。以独立运行目录保存前者的 final，再作为后者的 `--init-checkpoint`。脚本也生成其他消融轴，按计划选择，不必全跑；不同轴重复的默认配置去重。

`--init-checkpoint` 仅初始化权重并继承数据来源记录；`--resume` 才是恢复 optimizer/scheduler/RNG，必须保持模型、阶段、数据、world size、batch 和调度契约。详情见 [MODEL_CONTRACT.md](MODEL_CONTRACT.md)。

## 6. 评测、系统测量与验收

- Video-MME 固定帧数、分辨率/池化、循环次数、greedy decoding 和最大输出长度；多选默认可用 `--max-new-tokens 16`。严格解析失败计错。
- `--max-samples` 只是子集检查，不是全量成绩。记录全部覆盖率与每类数量；MVBench 任务宏平均与全部 QA 微平均不能混淆，当前分析输出不自动替代官方 aggregate。
- 多选准确率和开放回答 EM 分开报告；EM 不是语义 judge。每个 seed 保存单独预测和统计，再报告跨 seed 的均值/波动；现有脚本一次处理一个文件/一对文件，不自动汇总三 seed。
- 置信区间按原视频成簇 bootstrap；配对结果要求样本、来源、split、答案一致。
- H20/B300 使用相同模型、输入、精度和 batch，记录软件版本。保持相同步数和 global batch；另做等计算预算对照，不能凭峰值算力推断加速。
- 真实 checkpoint 还需 cache/no-cache 输出一致性、各参数组梯度和保存重载检查；目前这套严格自动测试是在 tiny CPU 模型上完成的，不应误写成 GPU 已测。
- 当前评测耗时是 **cold/mixed**，profile 样本额外有开销。正式速度表仍需专门 warmup/重复/同步测量；现有 CLI 不分别测 vision/prefill/decode，也没有完整 warmed benchmark runner。

实际排期只能在 pilot 后估算：`剩余优化步数 × 稳态秒/步 + 验证 + 检查点/I/O 时间`。首次编译、加载、下载、抽帧另记。H20 和 B300 各自测量，不预填小时数。

## 7. 每个实验必须交付什么

| 类别 | 必留记录 |
|---|---|
| 配置/版本 | 完整训练 JSON、Git commit、依赖版本、模型 SHA、manifest SHA、seed |
| 数据 | QA/来源视频数、split 和排除集合、覆盖率、缓存/抽帧参数 |
| 训练 | run_manifest、loss/验证日志、实际 global batch、稳定秒/步、异常与中断情况 |
| 检查点 | 中途可续训 checkpoint 与 final；对应来源记录；保存位置/校验信息 |
| 评测 | 逐样本 predictions JSONL、各指标 summary、配对/成簇统计、失败列表 |
| 系统 | preflight、GPU 型号/驱动、峰值显存、测量边界、warmup/重复策略（如已实现） |

`data/`、`weights/`、`checkpoints/`、`runs/` 默认在 experiment 下被忽略，**不是云备份**。自行将大检查点与完整日志保存到可靠存储；不要以为 push 代码就同步了它们。论文只同步经过审查的汇总表、真实结果图及其生成依据，不上传数据集原视频、密钥、绝对个人路径或未经授权的媒体。

完整细节入口：[上机命令](../README.md) · [数据](DATASETS.md) · [硬件](HARDWARE.md) · [消融设计](EXPERIMENTS.md) · [已验证与未验证](VALIDATION.md) · [论文](../../paper/README.md)。
