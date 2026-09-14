# 名词解释：FA 算子学习概念地图

这里汇总截至 **2026-09-14** 从可访问学习对话、已有学习仓库和本地概念笔记中整理出的概念，按“先理解数据，再理解模型和数学，最后理解硬件实现与验证”组织。

**阅读入口：[全部术语索引](index.md) · [概念关联图](relations.md) · [待核对问题](open-questions.md) · [资料覆盖与版本](sources.md)**

项目“资料”中的独立附件尚未成功下载，因此这是当前可访问材料的整理版，不能保证覆盖未读取的附件。词条出现代表已经讨论或在学习路线中提及，不表示已经掌握或完成上板验证。

## 按知识网络学习

先读 [知识网络与分层学习地图](knowledge-network.md)，按“前置知识 → 层内分组 → 跨层依赖 → 练习验收”推进。最近的硬件、分核与性能问题可按其中六次聚焦学习逐步连通；262 个术语继续使用现有词条。

## 八层目录

| 层级 | 解决的问题 | 阅读入口 |
|---|---|---|
| 1. 张量与代码 | 每个数、轴、操作和代码结构是什么？ | [Tensor、Shape、Stride、PyTorch 与 C++](01-foundations/tensors-and-code.md) |
| 2. 模型与推理 | 从文本到 Attention，再到生成怎样衔接？ | [Transformer、Prefill、Decode、KV Cache](02-model/transformer-and-inference.md) |
| 3. Attention 数学 | Q/K/V 如何得到输出，FA 怎样分块？ | [Attention / FA](03-attention/attention-and-fa.md) · [Online Softmax / expMax](03-attention/online-softmax.md) · [LSE](03-attention/LSE.md) |
| 4. 布局与寻址 | 数据各轴怎样组织，逻辑位置如何变成地址？ | [Layout](04-layout/layout.md) · [BNSD](04-layout/BNSD.md) · [BSND](04-layout/BSND.md) · [TND](04-layout/TND.md) |
| 5. 硬件与流水 | 谁计算，数据放哪里，为什么需要同步？ | [AIC/AIV、Cube/Vector、GM/L1/L0/UB、VF](05-hardware/ascend.md) |
| 6. 算子实现 | Host 怎样准备，Kernel 怎样消费？ | [调用层级](06-implementation/host-kernel.md) · [InferShape](06-implementation/InferShape.md) · [Tiling](06-implementation/Tiling.md) · [TilingData](06-implementation/TilingData.md) · [TilingKey](06-implementation/TilingKey.md) · [BNS1 分核与 Matmul](06-implementation/scheduling-and-matmul.md) |
| 7. 推理特性与具体算子 | 分页、稀疏、量化分别改变什么？ | [PA / Sparse / blockTable](07-features/paging-sparsity.md) · [量化 / MXFP8 / 精度](07-features/quantization.md) · [FA / FIA / QBSA / StemIndexer / MLA](07-features/operator-map.md) |
| 8. 验证与性能 | 如何判断正确、够准和更快？ | [测试、环境与性能瓶颈](08-validation/performance-and-tests.md) · [TND→BNSD/BSND 综合案例](08-validation/layout-case.md) |

层级是学习顺序，不是全部概念严格的继承树。例如 Layout 同时影响形状推导、Tiling 和 Kernel；GQA、PA、量化可同时出现在一个 case 中。跨层关系见[关联图](relations.md)。

## 最近问题直达

| 你问过的问题 | 对应记录 |
|---|---|
| BNSD、BSND、TND 有什么区别？ | [Layout 与三种轴顺序](04-layout/layout.md) |
| infershape 是什么？ | [输出形状推导](06-implementation/InferShape.md) |
| tilling 是什么意思？ | [正确拼写 Tiling：分块和分工](06-implementation/Tiling.md) |
| tilingdata、tilingkey 分别是什么？ | [执行参数](06-implementation/TilingData.md) · [实现选择标识](06-implementation/TilingKey.md) |
| lse 是什么？ | [每个 Query/head 的对数归一化统计量](03-attention/LSE.md) |
| 为什么分块后要改 expMax？ | [历史 max、sum、输出累积的换基](03-attention/online-softmax.md) |
| L1、UB、AIC、AIV、VF 是什么？ | [计算、存储与代码函数的层级](05-hardware/ascend.md) |
| BNS1 分核是什么？ | [任务数、核数、S1/S2 分块](06-implementation/scheduling-and-matmul.md) |
| PA、sparse 映射、blockTable 是什么？ | [从逻辑选块到物理地址](07-features/paging-sparsity.md) |
| MatmulBase、MatmulFull、模板参数？ | [具体源码中的执行路径](06-implementation/scheduling-and-matmul.md) |
| Cube bound、Vector bound、阻塞在 scale？ | [瓶颈与等待的区别](08-validation/performance-and-tests.md) |
| QBSA-MXFP8、Stem Indexer、FIA 是否同级？ | [算法、算子、路径和上游选块](07-features/operator-map.md) |

## 如何继续记录

遇到新词时先在索引查找，确定它属于哪一层；按[概念模板](template.md)记录“一句话、例子、关系、误区、证据”。已经存在的词优先更新原词条，把新增关系加到关系页。实现结论固定源码 commit，不把设计建议写成已支持的能力。

已有完整课程继续保留：[第 1 章](../01-llm-basics/README.md) · [第 2 章](../02-transformer-end-to-end/README.md) · [学习路线](../roadmap.md)。
