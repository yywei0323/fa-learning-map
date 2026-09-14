# InferShape（形状推导）

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 英文名：Infer Shape / Shape Inference
- 学习状态：待理解
- 首次记录：2026-09-10
- 最近更新：2026-09-14
- 原始问题：infershape 是什么？

## 一句话解释

**InferShape 根据算子的输入形状、属性和计算规则，推导输出张量有几个维度、每个维度有多大。**这里的 infer 表示“按规则推导”。

它回答“结果的 shape 是多少”，实际 Kernel 计算则回答“结果中每个数是多少”。框架可以利用形状信息连接后续算子、检查形状兼容性，并配合数据类型等信息准备输出内存。

## 具体例子：还没乘矩阵，也知道结果多大

```text
A 的 shape：(2,3)
B 的 shape：(3,4)
计算规则：C = A @ B

InferShape：检查内侧维度 3 与 3 匹配，推出 C 的 shape 为 (2,4)。
Kernel：执行矩阵乘法，算出 C 中的 8 个数。
```

这是普通二维矩阵乘法的教学例子。输出形状由规则确定，不需要先完成矩阵乘法。

## 在当前算子中做了什么

实现见 [quant_block_sparse_attn_infershape.cpp](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_infershape.cpp) 中的 `InferShapeQuantBlockSparseAttn`。以下以 2026-09-10 的本地 MXFP8 路径为准，输入为合法的 TND，且当前约束为 `D=D_v=128`。

假设 Query 的 shape 是 `(8,8,128)`，即 T=8、N1=8、D=128：

| 输出 | 推导出的 shape | 为什么 |
| --- | --- | --- |
| attention_out | `(8,8,128)` | 当前路径为每个 Query token、每个头输出一个 128 维向量 |
| softmax_lse，开启返回时 | `(8,8)` | 每个 token、每个 Query 头对应一个 LSE 统计值 |
| softmax_lse，关闭返回时 | `(0,)` | 当前 InferShape 用一维零长度形状描述该输出 |

这里的维度来自输入 shape 与属性；推导过程并没有计算 attention 或 LSE 的实际数值。`(0,)` 表示有一个轴且长度为 0，不是标量。

按代码可依次看到：

1. `GetInputShape`：读取 Query 的形状描述。
2. `GetAttrs`：读取 `layout_q`、`return_softmax_lse`、`quant_mode` 等属性。
3. `GetOutputShape`：取得可填写的输出形状描述。
4. `SetDimNum`、`SetDim` 或形状赋值：写入输出的维数和各轴长度。
5. 返回 `GRAPH_SUCCESS` 或失败状态。

函数末尾通过 `IMPL_OP_INFERSHAPE(QuantBlockSparseAttn).InferShape(...)` 注册回调，使框架能够找到这套推导规则。`GetOutputShape` 和 `SetDim` 操作的是形状描述，输出数据内存由相应框架或接入层准备。

这份公共函数也处理 NTD 输入，并将 attention_out 的形状排列成 TND。MXFP8 路径在 [Host 属性检查](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_check.cpp) 中进一步限制 Query 和输出都必须为 TND；不能只看公共 InferShape 分支来判断整条执行路径的支持范围。

## 与 InferDataType、Tiling、Kernel 的区别

| 名称 | 主要回答什么 | 本算子中的例子 |
| --- | --- | --- |
| InferShape | 输出有多大？ | attention_out 为 `(T,N1,128)` |
| InferDataType | 输出用什么数据类型？ | attention_out 为 BF16，LSE 为 FP32 |
| Tiling | 这次任务怎样分块、分配给核？ | 准备核数、分块和执行参数 |
| Kernel | 输出中的数具体是多少？ | 在 Device 上完成 attention 的计算与写回 |

同文件中的 `InferDataTypeQuantBlockSparseAttn` 独立设置输出类型。输出 shape 一样，不代表输入输出 dtype 也一样。

这些是职责上的区分，不是一张适用于所有运行模式的固定调用顺序表。当前仓库 [AI Core 开发指南](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/develop/aicore_develop_guide.md) 将形状推导放在 Host 侧，并介绍其图模式注册方式；Host、Device 与 Tiling 的说明见 [已有笔记](host-kernel.md)。

## 与 TND、BNSD、BSND 的关联

InferShape 必须按 layout 理解每个轴：TND 的第 0 轴是 T；BNSD 的第 0 轴是 B，第 2 轴才是 S；BSND 的第 1 轴是 S。

如果未来扩展输入输出布局，推导规则也必须同步调整。例如约定输出跟随 Query 布局时，可分别考虑 `(B,N,S,D_v)` 或 `(B,S,N,D_v)`。这是设计示例，当前实现的 Query 检查只接受三维 TND/NTD，尚未支持这里的四维输入。

修改输出 shape 描述本身不会转置数据；实际内存中的轴顺序还必须由数据转换或 Kernel 的寻址、写回实现保证。

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [Tiling](Tiling.md) | 对比 | InferShape 决定输出形状；Tiling 准备怎样分块执行 | 已确认 |
| [LSE](../03-attention/LSE.md) | 配合 | 推导 LSE 的输出形状；LSE 的实际数值由 Kernel 计算 | 已确认 |
| [layout](../04-layout/layout.md) | 使用 / 依赖 | 根据布局解释输入的轴，再推导输出形状 | 已确认 |
| [TND](../04-layout/TND.md) | 使用 / 依赖 | 当前 MXFP8 形状推导按 TND 解释 Query 的 T、N、D，再填写 attention_out 和 LSE 的形状 | 已确认 |

## 容易混淆的边界

- InferShape 不保证输出永远与输入同形。矩阵乘法、归约等算子需要各自的推导规则。
- 形状推导通常使用输入 shape 和属性；某些算子还依赖输入张量中的值。例如 Reshape 的目标形状如果作为一个输入张量传入，就需要读取它的内容。动态形状场景也可能需要运行时信息才能确定全部维度。
- 本算子的 TND 输出总大小可直接从 Query 的 T 获得，不必为此读取每条序列的长度；实际 Kernel 仍需通过长度信息区分各条序列。

## 参考来源

- [本地 InferShape / InferDataType 实现](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_infershape.cpp)：输出形状、类型及注册方式。
- [本地 MXFP8 接口文档](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)：TND 输入输出和当前 D 约束。
- [昇腾官方：InferShapeContext 简介](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/basicdataapi/atlasopapi_07_00094.html)：形状推导上下文，以及数据依赖场景的输入读取接口。
- [昇腾官方：算子入图基本开发流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0078.html)：数据依赖形状推导及 Reshape 示例。
