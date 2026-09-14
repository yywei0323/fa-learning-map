# 算法、算子与特性：FA / FIA / QBSA / MXFP8

[概念目录](../README.md) · [分页与稀疏](paging-sparsity.md)

## 它们不是同一层级

| 名称 | 所在层级 | 主要职责 |
|---|---|---|
| Attention | 数学运算 | 用 Q/K 得到权重，对 V 加权汇总 |
| FlashAttention / FA | 算法和实现方法家族 | 分块、在线 Softmax、减少中间矩阵的内存 IO |
| FIA / FusedInferAttentionScore | 昇腾推理融合 Attention 算子接口与实现 | 接收 Q/K/V 及推理特性参数，计算 Attention 输出 |
| PFA / PromptFlashAttention | 具体算子系列 | 侧重预填充计算 |
| IFA / IncreFlashAttention | 具体算子系列 | 侧重增量解码计算 |
| FlashAttentionScore / flash_attn | 源码中的具体算子或目录名 | 需查各自原型、入口和实现，不能与 FIA 调用链混用 |
| QBSA / QuantBlockSparseAttn | 具体分块稀疏 Attention 算子 | 显式消费稀疏块索引并完成量化相关计算 |
| QBSA-MXFP8 | QBSA 的特定量化执行路径 | 本快照用 quant_mode=2 选择 |
| StemIndexer | 稀疏 Attention 前处理算子 | 产生供下游消费的块选择结果 |
| FIA V3、V4、V5 | 接口版本语境 | 用对应文档、编译产物和参数签名理解 |
| FA1、FA2、FA3 | 算法/实现演进语境 | 编号不对应 FIA 接口版本 |

“FIA 是 FA 在昇腾上的推理算子实现”可以作为入口直觉，但 FIA 的名字不能保证每种 shape、特性、硬件都使用同一个模板，也不能推出其他 Attention 算子只是它的别名。

本仓库同时存在正式 `attention/fused_infer_attention_score` 与 `experimental/attention/fused_infer_attention_score` 目录，后者的示例范围不能套用前者的完整能力。

## 特性是不同的配置维度

| 特性 | 改变什么 | 主要关联 |
|---|---|---|
| MHA / GQA / MQA | Q 与 KV 的头组织 | 头映射、KV 复用 |
| BNSD / BSND / TND | 轴顺序、样本组织 | shape、stride、变长边界 |
| Prefill / Decode | 当前输入与缓存使用方式 | Sq/Skv、并行度 |
| PA | KV 存储和地址映射 | blockTable、page size |
| Sparse | 参与 Attention 的有效位置/块 | sparse_indices、mask、选块 |
| Quant / MXFP8 | 数值表示、scale、部分计算路径 | 分组、反量化、误差 |
| MLA | 压缩潜在表示与解耦位置成分等注意力架构 | 改变缓存表示和维度，不能直接套普通 GQA 缓存公式 |

MLA、SoftmaxFlash 接口差异、FIA V3 上板用例目前属于后续深入目标，不因为术语出现在笔记里就标为已掌握。

## 本次 MXFP8 记录的具体约定

以下是基线学习快照的接口约定，不是所有 MXFP8 算子的共同输入规格：

- Query 为 TND，KV 为 PA_BNBD，当前 D=Dv=128。
- Q/K 数据使用 FP8 E4M3FN；Q/K 沿 D 每 32 元素共享一个 E8M0 descale。
- V 沿 S 每 32 元素共享一个 E8M0 descale，不能把 Q/K 的分组轴直接套给 V。
- q_descale 为 [T,N1,D/64,2]；末尾 2 表示两个相邻 32 元素分组的 scale 打包，量化组仍是 32。
- p_scale 是本路径的额外缩放参数；Softmax 的 γ、Q/K/V descale、P 的量化 scale、online softmax 的 expMax 各自职责不同。
- 输出为 BF16，LSE 为 FP32。“全量化路径”不表示所有中间变量与输出都为 FP8。

[量化精度推导](quantization.md)解释为什么 FP32 累加不能恢复输入量化已经丢掉的信息。[版本案例](../08-validation/layout-case.md)单独记录 BNSD/BSND 扩展，不将其写成基线已经支持的能力。

源码：[FIA 说明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/fused_infer_attention_score/README.md)、[MXFP8 说明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)。
