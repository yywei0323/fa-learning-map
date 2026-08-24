# 从 Transformer 到 FIA Kernel：第一张全景图

## 1. 先确定你究竟在学什么

你不是在孤立地学一个 C++ 文件，而是在理解下面这条纵向链路：

```text
大模型为什么要生成下一个 Token
        ↓
Transformer Decoder 怎样得到上下文表示
        ↓
Attention 怎样用 Q/K/V 聚合历史信息
        ↓
FlashAttention 怎样减少 Attention 的内存 IO
        ↓
推理阶段 Prefill / Decode 怎样使用 KV Cache
        ↓
FIA 怎样把多种推理 Attention 场景收敛到融合算子
        ↓
Host Tiling 怎样选择实现并切分工作
        ↓
Ascend C Kernel 怎样搬运、计算、同步和写回
        ↓
aclnn Fuzz / Debug Tool 怎样证明它正确
```

这条链任何一层断掉，读 FIA 都会变成“看见很多 Shape、模板和宏，却不知道它们为什么存在”。

## 2. 大模型基础的最小知识框架

### 2.1 Token 与 Embedding

文本先被 tokenizer 切成 Token ID。Embedding 表把离散 ID 映射成连续向量。Transformer 处理的是这些向量，而不是汉字或单词本身。

### 2.2 自回归生成

Decoder-only 大模型按如下方式生成：

1. 输入 Prompt；
2. 预测下一个 Token 的概率；
3. 选出一个 Token；
4. 把它加入上下文；
5. 再预测下一个 Token。

因此推理天然分成：

- **Prefill**：第一次并行处理完整 Prompt；
- **Decode**：之后通常一次生成一个或少量 Token。

### 2.3 KV Cache

每一层 Attention 都会产生 K 和 V。Decode 时，历史 Token 的 K/V 不会变化，因此可以缓存起来，避免每步重算。

代价是：

- KV Cache 占用大量显存；
- 每个 Decode 步骤都要读取历史 KV；
- 长上下文时，读 KV 的带宽可能成为瓶颈；
- 需要管理不同请求、不同长度和动态分配。

这正是 GQA、MQA、PA、量化 KV、MLA 等特性的重要背景。

## 3. Transformer Decoder 的最小结构

一个典型 Decoder Block 可以抽象为：

```text
输入
 ├─ Norm
 ├─ Self-Attention
 ├─ Residual Add
 ├─ Norm
 ├─ MLP / MoE
 └─ Residual Add
输出
```

Attention 内部：

```text
Hidden States
  ├─ Wq → Q
  ├─ Wk → K
  └─ Wv → V
        ↓
RoPE / 位置处理
        ↓
Score = QKᵀ × scale
        ↓
Mask + Softmax
        ↓
Output = P × V
        ↓
Wo 投影
```

FIA 主要覆盖中间的融合 Attention 计算，并不等于完整 Transformer Block。

## 4. 从 Attention 到 FlashAttention

标准 Attention 数学式：

```text
S = QKᵀ × scale
P = softmax(S + mask)
O = PV
```

若完整生成并反复读写 S、P，长序列下会产生巨大的全局内存访问。

FlashAttention 的核心思想：

- 将 Q、K、V 分块；
- 块内完成 BMM1、Mask、Softmax 更新、BMM2；
- 使用 Online Softmax 合并各 KV 块；
- 不将完整注意力分数矩阵长期写回 GM；
- 用流水和双缓冲重叠搬运与计算。

所以它是 **数学上仍为精确 Attention、实现上重排数据流**。

## 5. 从 FA 到推理 FIA

推理算子必须处理的不只是一个 FA 公式，还包括：

- Prefill 与 Decode；
- KV Cache；
- MHA / GQA / MQA；
- 连续或分页 KV；
- 多种 Layout；
- 变长序列；
- Mask / Sparse；
- 量化与反量化；
- MLA 等新型注意力表示；
- 多芯片代际与不同实现模板。

FIA（FusedInferAttentionScore）的价值，是通过统一接口接收这些参数，再由 Host 识别场景、选择 Tiling 与 Kernel 路径。

## 6. FIA 代码的四层视角

| 层 | 首要问题 | 典型内容 |
|---|---|---|
| 接口层 | 调用者传了什么？ | aclnn 参数、Tensor、属性、Workspace、Stream |
| Host 层 | 这个 Case 属于哪种场景？ | 校验、Shape/Layout 解析、特性识别、Tiling Key |
| Tiling 层 | 怎样切才能正确且高效？ | Core 切分、块大小、缓冲预算、尾块、TilingData |
| Kernel 层 | 每个 Core 实际做什么？ | 搬入、BMM1、Mask、Softmax、BMM2、累加、写回 |

第一次源码阅读只跟一条最简单路径。先忽略量化、PA、MLA、稀疏等分支。

## 7. Host Tiling 的本质

Tiling 是连接“任意输入”与“有限硬件资源”的桥梁。

Host 侧通常要回答：

1. 输入各维度代表什么？
2. 需要启动多少个 Core？
3. 每个 Core 负责哪些 Batch/Head/Query 块？
4. Q、K、V 一次处理多大？
5. UB、L1、Workspace 是否放得下？
6. 尾块和对齐如何处理？
7. 该 Case 应选择哪个 Kernel 模板？
8. 哪些信息必须写入 TilingData？

Tiling 不是简单的“把长度除以块大小”，它同时承担功能分流和性能策略选择。

## 8. Kernel 的主干数据流

忽略工程细节后，FIA Kernel 可以先看成：

```text
for 每个 Query 块:
    载入 Q
    初始化历史 max / sum / output accumulator

    for 每个有效 KV 块:
        载入 K
        score = Q × Kᵀ
        score = score × scale + mask
        更新 online softmax 状态

        载入 V
        block_output = exp(score - new_max) × V
        按新 max 重标定历史 output
        合并 block_output

    output = accumulator / sum
    写回 GM
```

真实代码会因为 Cube/Vector 分工、数据格式、同步、流水和特性路径被拆得更复杂，但主干不变。

## 9. 第一条源码阅读路径

进入官方 ops-transformer/attention/fused_infer_attention_score 后，按以下问题做笔记：

1. aclnn V3 接口有哪些输入、输出和属性？
2. 最简单的 BNSD、FP16、MHA Case 会走哪个 Host 分支？
3. 哪段代码生成 Tiling Key？
4. 对应 TilingData 结构包含哪些字段？
5. Kernel 入口怎样根据 Key 选择模板？
6. BMM1、Softmax、BMM2 分别在哪？
7. Query/KV 循环边界来自哪些 Tiling 字段？
8. 最终归一化在哪里完成？
9. 输出 Shape 与 Layout 怎样映射？
10. Golden 实现怎样计算？

## 10. 目前不要急着做的事

- 不要一上来阅读所有 Tiling Key；
- 不要同时研究 GQA、PA、MLA 和量化；
- 不要把成功编译等同于结果正确；
- 不要在没有固定版本的情况下照搬 API；
- 不要在未理解 Online Softmax 时深入性能优化。

当前最优先的里程碑是：

> 选一个普通、无量化、无 PA、无 MLA 的 FIA V3 用例，能解释它从 aclnn 参数到 Host Tiling，再到 Kernel 主干和 Golden 对比的全过程。
