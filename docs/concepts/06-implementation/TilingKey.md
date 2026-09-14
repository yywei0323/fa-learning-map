# TilingKey（Kernel 分支选择标识）

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 英文名：TilingKey / tiling key
- 学习状态：待理解
- 首次记录：2026-09-10
- 最近更新：2026-09-14
- 原始问题：TilingKey 是什么？为什么既有 TilingData，又要有 Key？

## 一句话解释

**TilingKey 是 Host 在 Tiling 阶段设置的一个标识，用来选择本次执行使用哪个 Kernel 实现分支或模板组合。**在当前实现中，若干模板选择条件被编码成一个整数 Key。

同一个算子可能为不同量化模式、数据布局或功能开关准备不同实现。Key 把 Host 的选择与对应的 Kernel 代码关联起来。

## 与 TilingData 的区别

| 名称 | 回答的问题 | 当前算子中的例子 |
| --- | --- | --- |
| TilingKey | 本次使用哪个实现组合？ | MXFP8、TND、分页 KV、causal mask、返回 LSE、S2=512 配置 |
| TilingData | 这个实现本次要处理多大、怎样寻址？ | T、B、头数、头维度、分页参数、stride 等 |

这是一种职责区分，并不意味着字段绝不重叠。当前代码的 layout、mask、LSE 开关等同时出现在 Key 的选择条件和 Data 的字段中，两边的设置必须一致。

## 当前 Key 是怎样生成的

Host 的 `CalcTilingKey` 调用 `GET_TPL_TILING_KEY`，输入条件包括：

| 条件 | 当前 MXFP8 路径的取值或含义 |
| --- | --- |
| `QKV_DTYPE` | FP8 E4M3FN 的模板类型标记 |
| `LAYOUT_T` | Query 的布局，当前为 TND |
| `KV_LAYOUT_T` | 当前为分页 PA_BNBD 对应的枚举 |
| `MASK_MODE` | 无 mask 或 causal mask |
| `RETURN_SOFTMAX_LSE` | 是否返回 LSE |
| `Config` | S1=128、S2=512、D=128、DV=128 的计算模板配置 |
| `QUANT_MODE` | MXFP8 全量化模式 |

这里列的是选择条件，不是某个 Key 的实际十进制值。编码由模板声明中的字段顺序和位宽等规则共同决定，不需要凭空给一个数字赋予含义。

注意源码中的 `QBSA_KV_LAYOUT_PA_BNSD` 名字对应 `QBSALayout::PA_BNBD`，不能据此认为普通四维 Query BNSD 已经得到支持。

## 如何连接到 Kernel

1. 模板声明文件用 `ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_SEL` 声明参数及允许的组合。
2. Host 计算 Key，并通过 `context_->SetTilingKey(tilingKey_)` 设置本次选择。
3. 对应 Kernel 模板组合承接这些条件。当前入口根据模板参数 `QUANT_MODE`，通过 `if constexpr` 分别进入 MXFP8 或 FP8 实现。
4. MXFP8 分支读取 MX 专用 TilingData，随后初始化并执行 Kernel。

`if constexpr` 的条件在对应模板实例编译时确定；本次运行通过 Key 选择匹配实现。它不要求在每个元素的计算过程中重新判断所有条件。

## 两个容易混淆的地方

**Key 不需要为每个输入 shape 都不同。**当前 `CalcTilingKey` 不直接使用 T、B、头数等具体大小。保持上述选择条件不变时，T 从 256 变到 512 可沿用同一个 Key，而 TilingData 的 `t1Size` 随之改变。改变 mask 或 LSE 开关则会改变其编码条件。这个结论由当前 Key 的入参推导，不能推广到所有算子。

**Config 的编号、计算 tile 大小、用户序列长度是三件事。**当前 S2=512 的配置枚举值是 1；1 是配置编号，512 是该模板沿 KV 方向的逻辑计算 tile 大小，均不表示每条用户序列必须恰好长 512。更长的序列需要按实现循环处理。

添加新布局时，也不能只修改一个 Key 数字。Host 检查、模板组合、TilingData、Kernel 寻址和输出形状都要支持相同约定。

## 关联概念

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [Tiling](Tiling.md) | 由其产生 | Tiling 根据输入条件选择并设置本次实现标识 | 已确认 |
| [TilingData](TilingData.md) | 配合 | Key 所选 Kernel 必须按匹配的结构理解本次参数 | 已确认 |
| [layout](../04-layout/layout.md) | 使用 / 依赖 | Query、KV 布局是当前 Key 的选择条件 | 已确认 |

## 参考来源

- [Host Key 生成与设置](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_tiling.cpp)：`CalcTilingKey`、`SetTilingKey`。
- [模板组合声明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/quant_block_sparse_attn_template_tiling_key.h)：字段位宽、布局映射、Config 编号和允许组合。
- [Kernel 入口](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/quant_block_sparse_attn.cpp)：模板参数、`if constexpr`、各路径的 TilingData 读取。
- [昇腾官方：Tiling 实现](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0031.html)：TilingKey 用于关联 Host 选择与 Kernel 分支。

本仓库实现核查日期：2026-09-10；以上是静态阅读结果。
