# 推荐系统低精度训练：技术洞察与厂商竞争格局

> 更新时间：2026-07-13  
> 范围：面向 CTR/CVR、广告排序、内容推荐和大规模 DLRM/LRM 训练，重点讨论 FP8、INT8/INT4 Embedding、低精度通信及厂商公开路线。

## 1. 执行摘要

推荐系统低精度训练不是把整个模型统一切换成 FP8 或 INT4，而是按照模型组件和系统瓶颈进行分层设计：

- Dense MLP、Cross Network、Transformer 等矩阵计算优先采用选择性 FP8。
- 超大规模 Embedding 优先采用 INT8/INT4 存储、热点高精度 Cache 和混合 bit-width。
- Embedding All-to-All、Gradient All-Reduce 可采用 FP8、INT8 或 INT4 通信量化。
- Optimizer state、Master Weight、Normalization、Softmax、Loss 和敏感任务头通常保留 BF16/FP32。

截至 2026 年 7 月，公开证据最完整的是 Meta：已经形成 Dense FP8、Embedding INT4/INT8、4-bit 通信和 TorchRec/FBGEMM 工程框架的完整链路。华为诺亚公开了真正降低训练期 Embedding 精度的 ALPT；腾讯 FiT 侧重按特征重要性自动搜索 bit-width；Intel 与高校联合推进 DQRM INT4 QAT 和 INT8 梯度通信。NVIDIA、Google、AMD、AWS 和 Intel Gaudi 主要提供硬件与通用低精度平台，公开的推荐专项方案相对较少。

## 2. DLRM 是什么

DLRM 全称 Deep Learning Recommendation Model，是 Meta 提出并开源的标准化深度推荐架构，也是推荐系统低精度、Embedding 和分布式通信研究中最常见的公开基准。

```text
稀疏特征 ──> Embedding Tables ──┐
                                ├──> 特征交互 ──> Top MLP ──> CTR/CVR
稠密特征 ──> Bottom MLP ────────┘
```

DLRM 包含四类核心组件：

1. **Embedding Tables**：把用户 ID、商品 ID、广告 ID、类别 ID 等离散特征映射为向量。它通常占推荐模型绝大部分参数，规模可达到数百 GB 或 TB。
2. **Bottom MLP**：处理价格、统计量、时长等连续特征。
3. **Feature Interaction**：学习用户、商品、场景和上下文之间的交互关系。
4. **Top MLP**：输出点击率、转化率等预测结果。

DLRM 不等于某家公司的完整线上模型，但能代表推荐训练的主要系统矛盾：超大 Embedding、稀疏随机访问、All-to-All 通信、Dense GEMM 和极严格的 AUC/LogLoss 质量约束。

## 3. 为什么推荐系统不能直接照搬 LLM 的低精度方案

### 3.1 计算结构不同

LLM 以规则的大矩阵乘为主，容易发挥 FP8 Tensor Core 峰值；推荐模型包含大量小 GEMM、Jagged Tensor、Embedding Lookup 和特征交互，量化、反量化、转置及 scale 计算的开销可能超过低精度计算收益。

### 3.2 参数结构不同

传统推荐模型 99% 以上的参数可能位于 Embedding 表。Embedding Lookup 本身不是标准 GEMM，降低 Dense 计算精度并不能解决主要的容量与带宽问题。

### 3.3 通信模式不同

Embedding 模型并行依赖 All-to-All，Dense 数据并行依赖 All-Reduce。训练瓶颈可能是网络和参数搬运，而不是 FLOPs。

### 3.4 质量容忍度更低

广告和推荐业务通常对相对 LogLoss、GAUC 和校准误差非常敏感。Meta LoKA 的公开材料指出，在其生产场景中，约 0.02% 的相对 LogLoss 变化就可能被视为显著。

## 4. 推荐系统合理的混合精度架构

| 训练组件 | 建议精度 | 主要目标 | 主要风险 |
|---|---|---|---|
| 大型 Dense MLP/Transformer | FP8 E4M3 | 提升矩阵计算吞吐、降低激活显存 | 异常值、量化开销 |
| 反向梯度 GEMM | FP8 E5M2 或 BF16 | 提供更大的梯度动态范围 | 下溢和训练不稳定 |
| 小 GEMM/特征交互 | BF16/TF32，按层验证 FP8 | 避免量化开销超过收益 | 峰值 FLOPs 不等于端到端收益 |
| 热点 Embedding | BF16/FP16/INT8 | 保留高频更新精度 | 占用高精度 Cache |
| 长尾 Embedding | INT8/INT4 | 降低容量和内存带宽 | 长尾和冷启动精度 |
| Embedding 输出通信 | FP8/INT8/INT4 | 减少 All-to-All 流量 | scale 和小包开销 |
| Embedding Gradient | INT8/INT4 | 减少参数同步流量 | 小梯度被截断 |
| Dense Gradient All-Reduce | FP8/INT8，激进时 INT4 | 降低分布式通信量 | 需要误差补偿 |
| Master Weight/Optimizer | FP32/BF16 | 保留微小更新和长期累积 | 训练内存不能按 4 bit 简单估计 |
| Norm/Softmax/Loss/任务头 | BF16/FP32 | 保证概率和多任务稳定性 | 低精度误差被放大 |

推荐的目标配方：

```text
Dense 前向：选择性 FP8 E4M3
Dense 反向：FP8 E5M2 或 BF16
热点 Embedding：BF16/INT8 Cache
长尾 Embedding：INT4/INT8
Embedding 输出：BF16
Embedding Gradient 通信：INT8/INT4
Norm、Loss、Optimizer：BF16/FP32
```

## 5. 厂商与代表工作的对应关系

| 厂商 | 代表工作/平台 | 低精度对象 | 是否真正降低训练精度 | 主要价值 |
|---|---|---|---|---|
| Meta | LoKA | Dense MLP、交互层 | 是，选择性 FP8 计算 | 生产级推荐 FP8 训练 |
| Meta | Training with Low-precision Embedding Tables | Embedding 参数 | 是，FP16 存储+随机舍入 | 降训练内存、提升吞吐 |
| Meta | Mixed-Precision Embedding Using a Cache | Embedding 参数 | 是，INT4/INT8 主表+FP Cache | 长尾 INT4 Embedding 训练 |
| Meta | Quantized Collective Communications | All-to-All/All-Reduce | 是，4-bit 通信 | 降分布式训练网络流量 |
| Meta | TorchRec/FBGEMM/TorchAO | Embedding、Jagged Tensor、Dense | 工程底座 | 分片、低精度 Kernel、FP8 API |
| 华为诺亚 | ALPT | CTR Embedding | 是，INT8 低精度参数训练 | 自适应 scale、训练内存压缩 |
| 腾讯 FiT | MPE | CTR Embedding | 否，训练保留全精度参数 | 自动搜索 0–6 bit，偏 QAT/部署压缩 |
| Intel+CMU/UCB/UT Austin | DQRM | DLRM Embedding、梯度 | 部分，INT4 QAT+INT8 通信 | 模型和通信压缩 |
| NVIDIA | Transformer Engine+Merlin/HugeCTR | Dense FP8/FP4、分布式 Embedding | 提供底层能力 | GPU、低精度 Kernel 和训练框架 |
| Google | TPU FP8+SparseCore | Dense、Embedding | 提供底层能力 | Dense/Sparse 硬件协同 |
| AMD | ROCm/Quark/FP8 | Dense FP8 | 提供底层能力 | LoKA 已验证 MI300X/MI350X |
| AWS | Trainium2/Neuron | Dense FP8 | 提供底层能力 | FP8、随机舍入和专用通信 Core |
| 字节跳动 | Primus | 大规模 DLRM 训练 | 未公开低精度细节 | 推荐训练系统与混合流批训练 |
| 阿里 | RecIS | 超大规模稀疏模型 | 未公开低精度细节 | 推荐基础设施和 Embedding 管理 |

## 6. Meta：公开路线最完整

### 6.1 LoKA：生产推荐模型的选择性 FP8

LoKA 由 Meta AI 提出，并被 ISCA 2026 接收。其目标不是统一把 Linear 换成 FP8，而是建立推荐模型专用的逐算子决策框架：

- **LoKA Probe**：采集真实训练数据下的 Weight/Activation 分布，衡量每层 FP8 误差和 Kernel 性能。
- **LoKA Mods**：调整归一化、激活函数和模型结构，使敏感算子适配低精度并提高硬件效率。
- **LoKA Dispatch**：跨 TorchAO、FBGEMM、Transformer Engine、AMD/NVIDIA Kernel 选择满足误差阈值的最快实现。

Meta 报告的生产结果：

- 训练吞吐最高提升 20%；
- 推理最高提升 40%；
- 无质量损失；
- 覆盖 H100、B200、GB200、MI300X 和 MI350X；
- 已部署于服务数十亿用户的广告推荐模型。

LoKA 同时报告：直接使用通用 TorchAO FP8 配方时，代表性推荐模型可能出现约 1.3 倍变慢和最高 2.5% 相对 LogLoss 退化。因此推荐 FP8 的壁垒在于真实分布探测、模型改造和逐算子调度，而不仅是 FP8 Kernel。

### 6.2 低精度 Embedding 参数训练

Meta 2018 年的工作将 Embedding 参数存为 FP16，SGD 更新使用 FP32 计算，并通过 stochastic rounding 写回 FP16。公开结果包括最高约 2 倍内存节省和约 1.2 倍训练加速。

这条路线的核心原则是：

> Embedding 可以低精度存储，但更新计算和微小增量累积必须保留高精度路径或采用随机舍入。

### 6.3 INT4/INT8 主表+全精度热点 Cache

Mixed-Precision Embedding Using a Cache 利用推荐访问的 Zipf 分布：

- 大量长尾 Embedding 以 INT4/INT8 存储和训练；
- 约 1%–5% 的热门或近期访问行进入全精度 Cache；
- 低精度写回使用 stochastic rounding；
- 热点随访问频率动态换入换出。

公开结果：

- Criteo DLRM：INT8 主表+5% FP Cache，约 3 倍内存压缩；
- 工业模型：INT4 主表+1% FP Cache，超过 7 倍内存压缩；
- 端到端训练最高提升约 16%，主要来自减少 Host 到 GPU 的数据搬运。

### 6.4 4-bit 分布式通信

Meta 的 Quantized Collective Communications 针对：

- Embedding 前向 All-to-All；
- Embedding 反向 All-to-All；
- Dense Gradient All-Reduce。

论文显示上述通信可量化到 4 bit，同时维持其 DLRM 实验精度。实际收益依赖量化/反量化融合、消息大小、网络带宽和通信计算重叠。

### 6.5 Meta 路线总结

```text
Dense：选择性 FP8
热点 Embedding：FP16/BF16/INT8
长尾 Embedding：INT4/INT8
Embedding 更新：FP32 计算+随机舍入写回
All-to-All/All-Reduce：4-bit/8-bit
Optimizer：FP32/BF16
框架：TorchRec+FBGEMM+TorchAO
```

## 7. 华为诺亚：ALPT 低精度 Embedding 训练

华为诺亚方舟实验室与华中科技大学、清华大学提出 ALPT。它区别于只在前向插入 fake quant 的 QAT：Embedding 在训练期间以低精度格式保存，因此能够真正降低训练内存。

技术路线：

- 低精度存储 Embedding；
- 前向时反量化；
- 使用较高精度计算梯度和更新；
- 为不同特征自适应学习量化步长；
- 使用 stochastic rounding 写回低精度参数。

公开结论：

- 在 CTR 数据集上实现无损 INT8 Embedding 训练；
- 训练内存约压缩 3.2 倍；
- INT4/INT2 优于普通低精度训练，但没有证明 INT4 完全无损。

竞争定位：华为的公开特色是自适应量化步长和随机舍入的真正低精度 Embedding 训练，成熟口径主要是 INT8，而非 INT4。

## 8. 腾讯 FiT：MPE 自动混合 bit-width

MPE 由腾讯 FiT 与华中科技大学合作提出。它按照特征频率分组，在候选集合 `{0,1,2,3,4,5,6}` 中学习每组 bit-width 的概率分布，再依据容量正则项平衡 AUC 和模型大小。

需要准确区分：

- 训练阶段仍维护全精度 Embedding 参数；
- 通过 fake quant 和 STE 学习精度策略；
- 搜索后按采样 bit-width 重新训练；
- 部署时才以混合低精度格式存储。

因此，MPE 属于混合精度 QAT/模型压缩，不是训练内存和训练计算都降到 INT4。其优势是为不同频率和重要性的特征自动分配精度；0 bit 实质上等价于特征裁剪。

## 9. Intel 联合路线：DQRM INT4 QAT

DQRM 作者来自 CMU、UC Berkeley、Intel 和 UT Austin，其中 Intel 有共同作者，并使用 Intel oneCCL 和 Intel CPU 集群等环境。

技术路线：

- DLRM Embedding 做 INT4 QAT；
- 每次只复制和量化当前 batch 实际访问的 Embedding 行；
- 量化 scale 周期更新，避免每步扫描整张超大表；
- Embedding Gradient 稀疏化后量化为 INT8；
- Dense MLP 更敏感，可保留 FP32；
- 只为敏感 Dense 梯度维护 error compensation，避免给大表增加巨型误差缓存。

公开结果：

- Kaggle 模型从约 2.16 GB 降至 0.27 GB；
- Terabyte 模型从约 12.58 GB 降至 1.57 GB；
- 公开数据集精度与其 FP32 基线持平或略高。

但 DQRM 不应被宣传成单机 INT4 训练加速：

| 数据集 | FP32 时间/iter | DQRM INT4 时间/iter |
|---|---:|---:|
| Kaggle | 7 ms | 22 ms |
| Terabyte | 19 ms | 29 ms |

它的主要价值是 INT4 模型压缩、QAT 保精度和 INT8 梯度通信，而不是单机训练吞吐提升。

## 10. 硬件与平台厂商的准确定位

### 10.1 NVIDIA

NVIDIA 提供 H100/H200 的 FP8 E4M3/E5M2、Blackwell 的 MXFP8/NVFP4、Transformer Engine、cuBLASLt/CUTLASS，以及 Merlin/HugeCTR 分布式 Embedding 平台。它是底层低精度能力的主要提供者，但公开材料中缺少与 LoKA 对等的推荐专项逐层精度决策方案。

需要注意：Blackwell 的 4-bit 训练路线主要是 NVFP4，不是 INT4。

### 10.2 Google

Google 的优势是 TPU MXU 与 SparseCore 协同：Dense 使用 FP8，Embedding 和不规则稀疏访问交给 SparseCore。公开资料更多是硬件和通用 FP8/FP4 能力，推荐专项量化算法披露较少。

### 10.3 AMD

AMD MI300X/MI350X 支持 FP8，ROCm/Quark 提供通用量化能力。LoKA 已在这些硬件上完成验证，但 LoKA 的方法归属 Meta，AMD 是被验证的硬件后端。

### 10.4 AWS Trainium

Trainium2 支持 FP8、随机舍入、NeuronLink 和专用通信 Core。其公开优势是价格性能和分布式训练底座，推荐专项 INT4 Embedding 训练公开证据较少。

### 10.5 字节跳动与阿里

字节 Primus 和阿里 RecIS 展示了生产级推荐训练与超大稀疏模型能力，但公开材料没有完整披露 FP8/INT4 训练配方、质量变化和端到端收益。竞争分析中应标注“公开证据不足”，而不是推断其没有相关能力。

## 11. 技术成熟度与竞争判断

评分范围 1–5，评价对象是推荐系统训练，不是 LLM 推理；“未披露”不等于没有能力。

| 厂商 | FP8 Dense 推荐训练 | INT4/INT8 Embedding 训练 | 低精度通信 | 公开生产证据 | 综合判断 |
|---|---:|---:|---:|---:|---|
| Meta | 5 | 5 | 5 | 5 | 公开路线最完整，形成全栈闭环 |
| 华为 | 2 | 4，主要 INT8 | 2 | 2 | Embedding LPT 有特色 |
| 腾讯 | 1 | 3，偏 QAT/精度搜索 | 1 | 1 | 自动 bit-width 强，训练加速证据弱 |
| Intel | 2 | 4，INT4 QAT | 4，INT8 Gradient | 2 | 模型压缩和 CPU/通信路线清晰 |
| NVIDIA | 5，底层平台 | 3 | 4 | 3 | 硬件领先，推荐专项依赖伙伴或客户 |
| Google | 4 | 3，SparseCore | 4 | 2 | Dense/Sparse 硬件协同强，公开算法少 |
| AMD | 4，底层平台 | 2 | 3 | 3，由 LoKA 验证 | FP8 后端快速追赶 |
| AWS | 4，底层平台 | 2 | 4 | 1 | 通用 FP8 强，推荐专项证据少 |
| 字节/阿里 | 未披露 | 未披露 | 未披露 | 推荐系统有生产部署 | 不宜直接量化评分 |

## 12. 建议的落地路线

### 阶段一：建立 BF16/FP32 基线

记录：

- AUC、GAUC、LogLoss、NE、Calibration；
- 各任务头和长尾人群指标；
- samples/s、step time、GPU 利用率、HBM 占用；
- Embedding Lookup、All-to-All、Dense GEMM 和输入流水线耗时。

### 阶段二：选择性 FP8 Dense

- 优先量化大 MLP、Transformer FFN 和较大 Linear；
- 前向使用 E4M3，反向从 E5M2/BF16 起步；
- Norm、Softmax、Loss、最后一层和敏感任务头保留 BF16/FP32；
- 建立逐算子误差评估和 BF16 自动回退。

### 阶段三：Embedding INT8

- 从低敏感长尾表开始；
- 保留 BF16/FP32 更新路径；
- 使用 per-table、per-row 或 per-group scale 对比；
- 同时尝试 INT8 Embedding Gradient 通信。

### 阶段四：热点高精度+长尾 INT4

- 根据访问频率、梯度方差和业务重要性划分热冷层级；
- 热点行保持 BF16/INT8，长尾行使用 INT4；
- 使用随机舍入降低小更新丢失；
- 提供 INT4→INT8→BF16 自动回退。

### 阶段五：通信量化

- 独立评估 All-to-All 和 All-Reduce；
- 将 quant/dequant 与 collective 融合；
- 分析 scale 元数据、小包延迟和通信计算重叠；
- 必要时加入 error feedback。

### 阶段六：FP4/更激进 INT4 探索

- Blackwell 环境优先验证 NVFP4，而不是强行追求 INT4 Dense；
- 优先选择大 Dense tower、序列推荐 Transformer 和生成式推荐模块；
- 只有在融合 Kernel 后端到端收益明确时进入生产。

## 13. 汇报可用结论

### 一句话结论

> 推荐系统低精度训练正在形成“Dense 选择性 FP8/FP4、Embedding 热冷分级 INT8/INT4、通信 FP8/INT8/INT4、关键状态 BF16/FP32”的分层体系；竞争焦点已经从数据格式本身转向逐层精度决策、Embedding 更新、量化通信和模型—硬件协同。

### 厂商竞争结论

> Meta 已形成 Dense FP8、Embedding INT4、4-bit 通信和推荐专用框架的完整公开链路；华为侧重真正降低训练期内存的 INT8 Embedding LPT，腾讯侧重按特征重要性自动搜索 bit-width，Intel 联合学术界侧重 INT4 DLRM QAT 与 INT8 梯度通信；NVIDIA、Google、AMD 和 AWS主要提供低精度硬件与通用训练底座。

## 14. 参考资料

1. Meta, LoKA: Low-precision Kernel Applications for Recommendation Models At Scale: <https://arxiv.org/abs/2605.10886>
2. Meta, TorchRec: <https://github.com/meta-pytorch/torchrec>
3. PyTorch, torchao Float8 Training API: <https://docs.pytorch.org/ao/stable/api_reference/api_ref_float8.html>
4. Meta, Training with Low-precision Embedding Tables: <https://ai.meta.com/research/publications/training-with-low-precision-embedding-tables/>
5. Meta, Mixed-Precision Embedding Using a Cache: <https://arxiv.org/abs/2010.11305>
6. Meta, Training DLRM with Quantized Collective Communications: <https://ai.meta.com/research/publications/training-deep-learning-recommendation-model-with-quantized-collective-communications/>
7. Huawei Noah's Ark Lab et al., Adaptive Low-Precision Training for Embeddings in CTR Prediction: <https://arxiv.org/abs/2212.05735>
8. Tencent FiT et al., Mixed-Precision Embeddings for Large-Scale Recommendation Models: <https://arxiv.org/abs/2409.20305>
9. CMU/UCB/Intel/UT Austin, DQRM: Deep Quantized Recommendation Models: <https://arxiv.org/abs/2410.20046>
10. NVIDIA, Transformer Engine Low-precision Training: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/index.html>
11. NVIDIA, MXFP8: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html>
12. NVIDIA, NVFP4: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html>
13. Google Cloud, TPU7x FP8 Performance Guide: <https://docs.cloud.google.com/tpu/docs/ironwood-performance>
14. AWS, Trainium2 Architecture: <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/about-neuron/arch/neuron-hardware/trainium2.html>
15. AMD ROCm, FP8 Training with Megatron-LM: <https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html>
16. ByteDance, Primus: <https://www.usenix.org/conference/atc25/presentation/shan-jixi>
17. Alibaba, RecIS: <https://github.com/alibaba/RecIS>

