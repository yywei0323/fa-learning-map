# LSE（Log-Sum-Exp）

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 英文名：Log-Sum-Exp，简称 LSE
- 学习状态：待理解
- 首次记录 / 最近更新：2026-09-10
- 原始问题：lse 是什么？它与 layout 有什么关系？

## 一句话解释

**LSE 是把一组分数分别取指数、相加，再取自然对数得到的一个数。**在 Attention 中，它是某个 Query token、某个头对应的一行分数的 Softmax 归一化统计量。

```text
LSE(x) = ln(exp(x0) + exp(x1) + ...)
```

`exp(x)` 表示 e 的 x 次方，`ln` 是以 e 为底的自然对数。这里的 x 是进入 Softmax 的分数，已经考虑相应缩放与屏蔽规则；求和范围是该行允许参与计算的 KV 位置。

## 用一组小数字理解

假设一行只有两个有效分数 `[1,2]`：

```text
先取指数：exp(1) ≈ 2.7183，exp(2) ≈ 7.3891
相加：    Z = exp(1) + exp(2) ≈ 10.1073
取对数：  LSE = ln(Z) ≈ 2.3133
```

Softmax 则用每个指数值除以 Z，得到两个权重，约为 `[0.2689,0.7311]`。

| 对象 | 在此例中是什么 |
| --- | --- |
| 原始分数 | 两个数 `[1,2]` |
| Softmax 权重 | 两个数 `[0.2689,0.7311]`，和为 1 |
| LSE | 一个数 `2.3133`，表示归一化分母的自然对数 |
| Attention 输出 | 继续用这些权重加权组合 V 后得到的向量 |

因此 LSE 与权重有关，但它本身不是概率，也不是最终的 Attention 输出向量。知道原始分数和 LSE 时，可以按 `p_j = exp(x_j - LSE)` 得到权重；只有一个 LSE 值不能恢复全部分数或权重。

## 为什么经常和 FlashAttention 一起出现

直接计算很大分数的指数容易溢出。对至少含一个有限有效分数的行，可用等价形式：

```text
m = max(x)
l = sum(exp(x - m))
LSE = m + ln(l)
```

FlashAttention 可以在分块过程中累计行最大值 m 与对应的指数和 l，最后形成 LSE。它提供了重建归一化权重等操作所需的统计信息，重建时还需要相应分数。分块更新过程见 [FA 与在线 Softmax 笔记](online-softmax.md)。这不意味着每一步都必须先显式生成 LSE，再计算 Attention。

## 当前算子的 LSE 和 layout

在本地 MXFP8 路径中，启用 `return_softmax_lse=True` 时：

```text
Query：         (T,N1,128)
attention_out： (T,N1,128)
softmax_lse：   (T,N1)，FP32
```

一个 Query token 的一个头，对多个 KV 位置产生一行分数，这一行汇总成一个 LSE。因此 T 个 Query token、N1 个头共有 `T×N1` 个 LSE。例如 T=10、N1=8，LSE shape 为 `(10,8)`，共 80 个数。

**LSE 是对分数的 KV 位置轴做 log-sum-exp，不是直接把 Query 的 D 轴求和。**它没有 D 轴，是因为每个 token、每个头只保存一个行统计值。

[layout](../04-layout/layout.md) 说明这些数按哪些轴排列。当前 MXFP8 的 LSE 是 token 在前、head 在后；同一接口的 FP8 路径则采用 `(N1,T)`，可在 InferShape 中对照。因此 LSE 的数学含义与其布局约定需要分别理解。

`return_softmax_lse=False` 时，不返回有效 LSE 数值；当前 InferShape 将这个输出描述为 `(0,)`。该函数只推导形状，实际数值由 Kernel 计算：`SoftmaxLseCopyOut` 将累计的 sum/max 交给 `ComputeLseOutputVF`，并按 token/head 地址写回。

上述数字示例针对有有效分数的行。没有有效 KV 的行涉及接口的特殊约定，不能直接套用这里的普通 Softmax 示例。

## 关联概念

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [layout](../04-layout/layout.md) | 使用 / 依赖 | 读取 LSE 张量时需要知道各轴含义，当前 MXFP8 为 `(T,N1)` | 已确认 |
| [InferShape](../06-implementation/InferShape.md) | 配合 | InferShape 确定 LSE 的输出形状，Kernel 负责实际数值 | 已确认 |

返回 LSE 的开关也参与当前 [TilingKey](../06-implementation/TilingKey.md) 的模板选择；参与选择的是开关，不是计算出来的 LSE 数值。

## 参考来源

- [PyTorch 官方：torch.logsumexp](https://docs.pytorch.org/docs/2.9/generated/torch.logsumexp.html)：数学定义及数值稳定实现说明。
- [本地接口文档](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)：`softmax_lse` 的含义、类型与有效返回形状。
- [InferShape 实现](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_infershape.cpp)：MXFP8 / FP8 的轴顺序，以及未开启返回时的形状。
- [MX Vector 实现](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/quant_block_sparse_attn_mx_block_vec.h)：`SoftmaxLseCopyOut` 与 LSE 写回。

实现核查日期：2026-09-10；数学示例为教学计算，未作为算子的上板测试。
