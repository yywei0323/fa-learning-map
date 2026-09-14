# 第 7 层：分页、稀疏与索引

[概念目录](../README.md) · [量化精度](quantization.md)

**PA 决定 KV 存在哪里；稀疏选择决定计算哪些 KV；Tiling 决定怎样分批计算。** 这三个维度可以组合，但彼此不能代替。

## PA、Page 与 blockTable

PA 通常指 PagedAttention。KV Cache 按固定页大小组织，每条序列的逻辑页可以映射到物理池中的不同页。

| 名词 | 含义 |
|---|---|
| Page / KV block | KV Cache 的分配、寻址单元 |
| Logical page | 某条序列中的页编号 |
| Physical page | 全局 KV 存储池中的实际页编号 |
| blockTable / block_table | 每个 batch 从逻辑页编号到物理页编号的映射 |
| pa_block_size | 一页容纳的 token 数 |
| PA_BNBD | 当前 QBSA 文档的分页 KV 布局名，实际 shape 为 [block_num,N2,pa_block_size,D] |
| cu_seqlens / seqused | 前缀边界或实际长度信息；具体含义、是否可省略按接口区分 |

PA_BNBD 中的 block_num 是物理页数，不能把它当普通 BNSD 的 batch。访问 token 150、页大小 128，先得到逻辑页 1、页内偏移 22，再查 `block_table[b,1]`。最后还要加 head、D 和实际 stride 才是数据地址。

## 稀疏化与 sparse 映射

Dense Attention 对所有允许的位置计算；Sparse Attention 只处理选中的一部分位置或块。相对同一个 dense 目标，删去原本有效的位置通常会改变数学结果。对所选集合内部仍须正确归一化。

| 名词 | 含义 |
|---|---|
| Block Sparse | 以块为单位选择、跳过 KV 区域 |
| sparse_indices | 每个 Query 块保存的逻辑 KV 块编号列表 |
| sparse_seq_len | 当前 Query 块实际有效的索引数量；不是一定等于 KV token 总长度 |
| Sparse mapping | 稀疏列表位置 → 逻辑 KV 块 → token 区间 → PA 页与页内地址 |
| Sparse density | 保留部分相对于候选总体的比例，必须说明按块还是按元素统计 |
| Empty sparse row | Query 块没有有效稀疏项，需要按接口处理输出与 LSE |
| Sink / Window / TopK blocks | 始终保留的起始块、局部窗口块、按得分选择的块；具体规则由选块算法定义 |

例如 sparse_indices 的第 1 项为 3，表示第 3 个逻辑 KV 稀疏块，不直接表示物理页 3。若稀疏块大小 128、PA 页大小 256，则该块起始 token 为 384，逻辑页为 1，页内偏移为 128。只有尺寸与对齐满足特定关系时，才能直接用 sparse_id 查页表。

数据和 scale 必须使用相同的逻辑索引映射；否则可能把一个块的数据乘上另一个块的缩放值，输出 shape 看起来仍然正确。

## 四种“块”应分开记录

| 分块对象 | 解决的问题 | 本学习场景中的例子 |
|---|---|---|
| FA tile | 运算和片上容量 | 128 行 Q、512 列 KV 的预定义计算配置 |
| Sparse block | 选择哪些 token 对参与 | 按一个逻辑 KV 区间选中或跳过 |
| PA page | KV 的存储分配与寻址 | 一个物理页存若干 token |
| Quant group | 哪些元素共享 scale | MXFP8 每 32 个元素共用一个 E8M0 scale |

这些大小相等时也只是实现选择，不是定义上的相等。减少到八分之一稀疏块不能直接推出八倍加速：选块、搬运、scale、尾块和负载均衡仍有成本。

## Stem Indexer 的上下游位置

本学习快照的 StemIndexer 根据 qflat/kflat 的块级相关性和 vbias 进行打分，再按动态 TopK 预算、Sink 与 Window 规则选择块，输出 sparse_indices 和 sparse_seq_len。QBSA 使用这些索引完成所选 KV 的 Attention。

qflat/kflat 是用于选块的压缩表示，vbias 是选块评分的 Value 量值偏置；它们不等于最终 Attention 的完整 Q/K/V 或 PV 输出。

这是具体算子接口的关系，不证明 FIA 内部一定调用 StemIndexer，也不意味着 Indexer 与 QBSA 采用相同 dtype。

源码：[StemIndexer 说明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/stem_indexer/docs/StemIndexer.md)、[QBSA-MX 接口](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)。
