# TilingData（执行参数数据）

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 英文名：TilingData / tiling data
- 学习状态：待理解
- 首次记录：2026-09-10
- 最近更新：2026-09-14
- 原始问题：TilingData 是什么？它与 TilingKey 有什么区别？

## 一句话解释

**TilingData 是 Host 在 Tiling 阶段准备、传给 Device Kernel 使用的一组执行参数，通常用结构体组织。**它可以包含任务规模、分块信息、步长、布局及其他执行配置。

先理解 [Tiling](Tiling.md)：大任务往往需要分给多个计算核，并在核内分块处理。Host 根据输入描述、算子属性和硬件条件准备执行方案；TilingData 是其中一项产物。Tiling 还可以单独设置 TilingKey、启动核数和 workspace 大小，不能把全部 Tiling 输出都等同于 TilingData。

## 一个简单的分块例子

假设处理 1000 个元素，每块最多 256 个，依次处理 `256、256、256、232` 个元素。一种教学用结构可以记录：

```text
totalLength = 1000
tileLength  = 256
tileCount   = 4
tailLength  = 232
```

这些字段是为了说明概念设计的示例，不是本算子的实际字段；也可以只传部分参数，让 Kernel 推导其余值。TilingData 保存的是执行参数，待处理的 1000 个数仍由输入张量承载。

## 本算子的真实 TilingData

当前 MXFP8 路径使用独立的 `QuantBlockSparseAttnMxTilingData`。部分真实字段如下：

| 字段 | 含义 |
| --- | --- |
| `baseParams.bSize` | batch 大小 B |
| `baseParams.t1Size` | Query 有效 token 总数 T |
| `baseParams.n2Size`、`gSize` | KV 头数与分组比例，Query 头数为 N2×G |
| `baseParams.dSize`、`dSizeV` | Q/K 头维度与 V 头维度 |
| `baseParams.coreNum` | 本次使用的核数参数 |
| `attrParams.layoutQ`、`baseParams.outputLayout` | Query 与输出的布局枚举 |
| `baseParams.keyStrides` 等 | 分页数据或量化 scale 的寻址步长 |
| `pageAttentionParams.blockSize` | PA 物理页的大小 |
| `pageAttentionParams.qBlockSize`、`kvBlockSize` | 稀疏 Q/KV 块的大小 |
| `scaleParams` | 与量化 scale 的形状、模式等有关的参数 |

这里的页大小、稀疏块大小有各自含义；不能全部理解成 Kernel 一次计算的 tile 大小。当前 MX 计算 tile 的 S2=512 由模板配置选择，见 [TilingKey](TilingKey.md)。

例如 B=2、T=256、Query 头数 8、KV 头数 2、D=128 时，部分字段为 `bSize=2、t1Size=256、n2Size=2、gSize=4、dSize=128`。这只是字段映射例子，不是一份完整的可执行调用参数。

## 从 Host 到 Kernel 的路径

1. `FillMxTilingData` 根据已解析的输入信息填充结构体。
2. `SaveTilingData` 在 MX 分支检查缓冲区容量，将结构体复制到 raw tiling buffer，并设置数据大小。
3. 执行时，Kernel 通过入口的 `tiling` 参数取得这份信息。
4. `GET_TILING_DATA_WITH_STRUCT(QuantBlockSparseAttnMxTilingData, ...)` 按 MX 结构解析，再传入 Kernel 的 `Init`。

Host 写入与 Kernel 读取的字段类型、顺序和结构必须一致。当前 FP8 和 MXFP8 路径使用不同结构，不能混读；修改结构定义时，需要同步检查填充和读取代码。

## 与 TilingKey 的区别及关联

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [Tiling](Tiling.md) | 由其产生 | Tiling 准备并保存本次执行参数，Kernel 按对应结构读取 | 已确认 |
| [TilingKey](TilingKey.md) | 配合 | Key 选择执行分支，Data 提供该分支本次执行需要的参数，两者必须匹配 | 已确认 |
| [layout](../04-layout/layout.md) | 使用 / 依赖 | 当前 Data 包含布局枚举，并携带相关寻址参数 | 已确认 |

同一个 Key 可以对应不同的 TilingData：例如布局、量化模式等选择条件不变，只改变 T，当前 `CalcTilingKey` 不会因 T 单独变化而产生新 Key，但 `t1Size` 会改变。这是根据当前代码作出的推导。

## 参考来源

- [MX TilingData 结构定义](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/quant_block_sparse_attn_mx_tiling_data.h)：实际字段。
- [Host Tiling 实现](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_tiling.cpp)：`FillMxTilingData`、`SaveTilingData`、`DoOpTiling`。
- [Kernel 入口](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/quant_block_sparse_attn.cpp)：按对应结构读取 TilingData。
- [昇腾官方：Tiling 实现](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0031.html)：TilingData、TilingKey、block_dim 与 workspace 的分工。

本仓库实现核查日期：2026-09-10；以上是静态阅读结果。
