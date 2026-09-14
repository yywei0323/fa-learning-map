# Tiling（分块与任务划分）

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 英文名：Tiling（一个 l；tile 表示小块）
- 学习状态：待理解
- 首次记录 / 最近更新：2026-09-14
- 原始问题：tilling 是什么意思？

## 一句话解释

**Tiling 是把大的计算任务划分为能分批处理的小块，并安排这些任务怎样交给计算核执行。**在本仓库的 Host Tiling 实现中，程序根据输入描述、算子属性和硬件条件，准备分块、核数及实现选择等执行参数。

一个小块称为 tile；决定怎样划分和处理的规则称为 Tiling 策略。Host 准备参数，Device Kernel 按匹配的策略搬运数据、计算并写出结果。

## 为什么要分块

AI Core 的片上缓冲区容量有限，大张量通常不能一次全部放进去计算。逐块搬入、计算、搬出，可以让大任务在有限空间内完成；把任务分给多个核，还可以利用并行计算能力。

分块策略还会考虑对齐、数据复用、搬运开销和负载均衡。块过小可能增加循环与搬运开销，块过大则可能超出可用缓冲区，因此大小要结合实际算子和硬件选择。

## 同时看懂“分给多个核”和“每核分块”

以逐元素加法 `y[i]=a[i]+b[i]` 为教学例子：总共 1000 个元素，有两个计算核，假设每次最多处理 256 个元素。

一种安排是每个核负责 500 个，再各自分成两次处理：

| 核 | 负责的全局下标 | 第一次 | 第二次 |
| --- | --- | --- | --- |
| 核 0 | 0～499 | 下标 0～255，共 256 个 | 下标 256～499，共 244 个 |
| 核 1 | 500～999 | 下标 500～755，共 256 个 | 下标 756～999，共 244 个 |

```text
1000 个元素
├─ 核 0：500 个 → 256 + 244
└─ 核 1：500 个 → 256 + 244
```

Tiling 要确定使用几个核、各核负责什么、每次处理多少，以及最后不足一整块时怎样处理。这里的 244 个就是尾块的有效元素数，Kernel 不能把超出有效范围的位置当成正常输入。

这组数字只演示分工；真实实现还需满足数据搬运与计算的对齐要求。一个核可以循环处理多个 tile，tile 的数量并不等于启动核数。

## 与已经学过的概念放在一起

| 概念 | 回答的问题 | 上述例子 |
| --- | --- | --- |
| InferShape | 输出张量有多大？ | 输出仍有 1000 个元素 |
| layout | 张量各轴代表什么、如何排列？ | 当前例子只有一个元素下标轴 |
| Tiling | 怎样把这次计算分块、分工？ | 两个核，各 500，每核分 256 与 244 两次 |
| TilingData | 将哪些具体参数传给 Kernel？ | 可以记录总长度、每核长度、块大小等 |
| TilingKey | 使用哪个 Kernel 实现分支或模板组合？ | 选择与本次类型、模式等条件匹配的实现 |
| Kernel | 每个输出数具体是多少？ | 搬入 a、b 的对应块，执行加法并写出 |

所以，Tiling 是准备执行方案的过程；TilingData 和 TilingKey 是它准备的信息。Tiling 还可以设置启动核数、workspace 大小等执行信息。上述职责对照并不规定所有运行模式下的固定调用顺序。

## 当前 MXFP8 算子里的实际代码

入口是 [TilingQuantBlockSparseAttn](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_tiling.cpp)。它先通过 `parser.Parse` 解析输入描述，再通过 `checker.Process` 检查，随后进入 `DoOpTiling`。

`DoOpTiling` 当前的主要动作是：

1. 根据 B、Query 头数和 Query 方向的块数等信息计算任务数。
2. 查询可用 AI Core 数，将 `usedCoreNum_` 设为硬件核数与任务数中的较小者。
3. MXFP8 路径调用 `FillMxTilingData`，填写本次的尺寸、布局、分页等参数。
4. 调用 `CalcTilingKey`，选择对应的量化模式、layout、mask、LSE 开关和计算模板组合。
5. 通过 `SetBlockDim`、`SetTilingKey` 设置执行信息，并由 `SaveTilingData` 保存参数数据。

这里并不是所有块大小都在运行时重新搜索得出。当前 MXFP8 路径选用预定义的 S1=128、S2=512 模板配置；S2=512 描述计算 tile，不能当作用户序列总长度必须等于 512。具体地址计算、块循环与任务执行仍要结合 Kernel 阅读。

## 容易混淆的地方

- **Tiling 与 layout：**layout 定义数据各轴的组织方式，Tiling 决定怎样组织分块计算。Tiling 会使用 layout 信息，但不会仅因选择了分块方案就自动完成布局转换。
- **核数与 tile 数：**一个核可处理多个 tile；最后一个 tile 也可能不足固定块大小。
- **计划与实际计算：**本项目的 Host Tiling 函数主要处理描述与执行参数，算子的主体数据计算由 Device Kernel 完成。

## 关联概念

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [TilingData](TilingData.md) | 产生 | 将本次尺寸、分块与寻址等执行参数组织成数据传给 Kernel | 已确认 |
| [TilingKey](TilingKey.md) | 产生 | 根据输入条件选择并设置匹配的实现标识 | 已确认 |
| [layout](../04-layout/layout.md) | 使用 / 依赖 | 按布局解释输入各轴，准备对应的执行参数与实现选择 | 已确认 |
| [InferShape](InferShape.md) | 对比 | InferShape 决定输出形状，Tiling 准备分块和执行方案 | 已确认 |

## 参考来源

- [仓库 AI Core 开发指南](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/develop/aicore_develop_guide.md)：Tiling 简介与 Kernel 分工。
- [本地 Host Tiling](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_tiling.cpp)：`TilingQuantBlockSparseAttn`、`DoOpTiling`、`CalcTilingKey`。
- [模板组合声明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/quant_block_sparse_attn_template_tiling_key.h)：当前 MXFP8 的预定义配置。
- [昇腾官方：Host 侧 Tiling 基本流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_0064.html)：根据 shape 等信息确定切分算法参数。

本地实现核查日期：2026-09-14；分块例子为教学示意，未作为上板测试。
