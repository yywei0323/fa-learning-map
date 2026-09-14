# layout（数据布局）

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 英文名：layout / data layout
- 学习状态：待理解
- 首次记录：2026-09-10
- 最近更新：2026-09-14
- 原始问题：layout 是什么？与 TND、BNSD、BSND、Tiling 有什么关系？

## 一句话解释

**layout 是数据布局。在这里，它约定注意力张量的各个轴代表什么、按什么顺序排列。**例如 BNSD 表示依次为 batch、注意力头、序列位置、头内向量分量。

可以把 layout 理解为“查找一个数时，坐标按什么顺序写”。例如 `Q[b,n,s,d]` 按“哪条序列、哪个头、哪个 token、向量里的哪个数”定位元素，这就是 BNSD。改成 `Q[b,s,n,d]` 就是 BSND。

shape 给出每个坐标可取的范围，layout 说明每个坐标的含义。例如 `(2,8,5,128)` 加上 BNSD，才能知道 8 是头数、5 是序列长度。

## shape、layout、stride、dtype 分别说明什么

| 名称 | 回答的问题 | 例子 |
| --- | --- | --- |
| shape | 每个轴有多大？ | `(2,8,5,128)` |
| layout | 这些轴分别是什么？ | BNSD：B=2、N=8、S=5、D=128 |
| stride | 沿某个轴走一步，内存跨过多少元素？ | 上述标准连续张量的 stride 为 `(5120,640,128,1)` |
| dtype | 每个元素用什么数值类型表示？ | BF16、FP32、FP8 |

stride 数值由各轴长度相乘推导，单位是元素；字节地址还要结合元素大小和存储起点。相同 shape、layout 的张量也可能有不同 stride，因此不能只看 layout 就判断任意张量的全部物理地址。

本接口中的 `FORMAT_ND` 是框架的数据格式标记；`layout_q="TND"` 等属性进一步说明注意力轴的业务含义。ND 中的字母不能直接当作 TND 中 N、D 两个轴来解释。

## 具体例子

两条序列都长 5，每个 token 有 8 个头，每个头有 128 个数：

| layout | shape | 一个元素的索引形式 |
| --- | --- | --- |
| [BNSD](BNSD.md) | `(2,8,5,128)` | `Q[b,n,s,d]` |
| [BSND](BSND.md) | `(2,5,8,128)` | `Q[b,s,n,d]` |
| [TND](TND.md) | `(10,8,128)` | `Q[t,n,d]`，此例 `t=b*5+s` |

三种布局可以表达同一份逻辑数据，但转换时要保持元素对应关系。仅更改 layout 字符串不会搬运或转置数据。变长 TND 还需保留序列边界，转为普通四维布局时通常需要补齐，详见 TND 笔记。

## 在本仓库中的使用

截至 2026-09-10 的本地实现：

1. Host 的 `ParseQuery` 根据 `layout_q` 识别轴，从输入 shape 中读出 T、N、D。
2. `InferShape` 按布局解释输入并确定输出形状。
3. `FillMxTilingData` 将布局枚举写入 `attrParams.layoutQ`、`baseParams.outputLayout` 等字段。
4. `CalcTilingKey` 把布局作为选择 Kernel 模板组合的条件之一。
5. Kernel 根据布局和相应参数寻址、计算与写回。

这表示各部分如何使用布局，不是所有运行模式都固定遵循的调用顺序。当前 MXFP8 路径的 Query 和输出仅支持 TND，KV 使用 PA_BNBD。

## 关联概念

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [Tiling](../06-implementation/Tiling.md) | 被使用 / 依赖 | Tiling 按布局理解各轴，准备分块、寻址和实现选择参数 | 已确认 |
| [LSE](../03-attention/LSE.md) | 被使用 / 依赖 | LSE 张量的 token/head 轴顺序由接口布局约定 | 已确认 |
| [BNSD](BNSD.md)、[BSND](BSND.md)、[TND](TND.md) | 包含具体类型 | 它们是注意力张量 layout 的具体取值 | 已确认 |
| [InferShape](../06-implementation/InferShape.md) | 被使用 / 依赖 | 形状推导需要知道各轴的含义 | 已确认 |
| [TilingData](../06-implementation/TilingData.md) | 被使用 / 依赖 | 当前参数结构中保存布局枚举，供执行时使用 | 已确认 |
| [TilingKey](../06-implementation/TilingKey.md) | 被使用 / 依赖 | 当前模板选择把布局作为编码条件之一 | 已确认 |

## LSE 也有自己的布局

[LSE](../03-attention/LSE.md) 是算出来的行统计量，layout 是张量的轴组织方式。当前 MXFP8 路径中，Query 的布局为 TND，而有效 LSE 的 shape 为 `(T,N1)`：每个 token、每个头各保存一个数。例如 `(10,8)` 表示 10 个 token、每个有 8 个头的统计量，共 80 个数。

读 `lse[t,n]` 时，t 选择 token，n 选择头。另一个接口若约定 `(N,T)`，就应按 `lse[n,t]` 读取同样语义的位置。因此需要同时知道一个结果的数学含义和它的轴排列。

## 参考来源

- [Host 输入解析](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_info_parser.cpp)：`ParseQuery`。
- [Host Tiling 实现](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_tiling.cpp)：布局字段与模板选择。
- [数据格式说明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/context/data_format.md)、[非连续张量说明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/context/non_contiguous_tensor.md)：format、shape、stride 与 offset。

## 其他已经遇到的写法

- **BSH**：`[B,S,H]`，隐藏特征维 H 尚未显式拆成 head。普通多头投影中 H=N*D，可以进一步组织为 BSND；是否要复制仍取决于 stride。
- **NTD**：`[N,T,D]`，与 TND 交换 N/T 轴。公共 InferShape 中存在该分支，不意味着每个量化路径都允许这种输入。
