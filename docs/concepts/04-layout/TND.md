# TND

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 学习状态：待理解
- 首次记录 / 最近更新：2026-09-10
- 原始问题：TND 是什么意思？为什么没有 B 和 S？

## 一句话解释

**TND 是 shape 为 `(T,N,D)` 的注意力张量布局：把一批序列的有效 token 按顺序拼接，T 是拼接后的 token 总数，N 是注意力头数，D 是每个头的向量维度。**

设第 b 条序列的有效长度为 `S_b`，则 `T=Σ_b S_b`。只有所有序列等长为 S 时，才可以直接写成 `T=B×S`。

B 和每条序列的长度仍然存在于问题中，但不再分别占用张量的两个轴；需要额外的边界或长度信息才能找回各条序列。

## 具体例子

第一条序列有 3 个 token，第二条有 5 个，每个 token 有 8 个头，每个头有 128 个数：

```text
序列 0：a0 a1 a2
序列 1：b0 b1 b2 b3 b4

T 轴： a0 a1 a2 b0 b1 b2 b3 b4
下标：  0  1  2  3  4  5  6  7

shape = (8,8,128)
cu_seqlens_q = [0,3,8]
```

shape 中第一个 8 是 `T=3+5`，第二个 8 是头数 N，两者只是恰好相等。

本算子的 `cu_seqlens_q` 是 Query 的累计长度表，包含开头的 0 和末尾的总数，因此有 `B+1` 项。它把下标 0、1、2 划给第一条序列，把 3、4、5、6、7 划给第二条序列。

```text
第 b 条序列的起点 = cu_seqlens_q[b]
第 b 条序列的长度 = cu_seqlens_q[b+1] - cu_seqlens_q[b]
其第 s 个 token 的全局下标 t = cu_seqlens_q[b] + s
```

下标从 0 开始，第二条序列的第 1 个 token 是 `Q[3,:,:]`。只有 `T=8`，无法知道原来的长度是 3、5 还是 4、4。

**拼接的是存储，序列边界仍需保留。**不能因为 token 放到同一个 T 轴，就让不同序列任意互相做注意力计算。

## 为什么用 TND

变长序列可以只存有效 token，避免每条都补齐到最长长度。上例按 S=5 补齐的四维布局有 `2×5=10` 个位置，TND 则有 8 个。这说明存储位置减少，不代表整体运行时间必然更短。

TND 也能存等长序列。两条都长 5 时，shape 为 `(10,8,128)`，边界为 `[0,5,10]`。

## 与其他布局的关系

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [layout](layout.md) | 属于 | 本概念是注意力张量数据布局的一种 | 已确认 |
| [InferShape](../06-implementation/InferShape.md) | 被使用 / 依赖 | 当前 MXFP8 输出形状推导依据 TND 的轴含义解释 Query | 已确认 |
| [BSND](BSND.md) | 对比 | 等长、无补齐且按 batch 顺序排列时，可合并 B、S 得到 T；变长时要按实际长度去掉补齐位置再拼接 | 已确认 |
| [BNSD](BNSD.md) | 对比 | 等长时可先交换 N、S 转成 BSND，再合并 B、S；一般不能直接修改 shape 得到 TND | 已确认 |

以上转换条件是按轴含义推导的。TND 反向恢复普通四维张量时，等长场景需知道 B、S；变长场景需按边界拆分并按约定补齐。

## 在本仓库中的含义与位置

截至 2026-09-10 的本地 MXFP8 路径，Query 为 `(T,N1,D)`，输出为 `(T,N1,D_v)`，其中 N1 是 Query 头数。Query 和输出当前只接受 TND。

[MX Kernel](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/quant_block_sparse_attn_mx_kernel.h) 在 batch 变化时读取 `cuSeqlensQGm` 的相邻两项，求出 `queryTokenBase` 和 `actualS1Size`。

KV 使用另一种分页布局 `PA_BNBD`，实际 KV 长度由 `seqused_kv` 记录。Query 是 TND，并不意味着 KV 也使用 TND。

这里的 `[0,3,8]` 是本算子的前缀表约定；其他接口的长度参数是否包含开头的 0，应查相应文档。

## 参考来源

- [本地 MXFP8 接口文档](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)：Query shape、布局限制、`cu_seqlens_q` 数值约定及分页 KV。
- [Host 属性检查](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_check.cpp)：MXFP8 路径要求 Query、输出均为 TND。
- [已有布局学习笔记](../07-features/operator-map.md)：布局对照、长度边界与转换推导。
