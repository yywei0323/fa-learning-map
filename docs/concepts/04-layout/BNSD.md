# BNSD

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 学习状态：待理解
- 首次记录 / 最近更新：2026-09-10
- 原始问题：BNSD 是什么意思？它和 TND、BSND 有什么关系？

## 一句话解释

**BNSD 是注意力张量的一种四维布局，shape 为 `(B,N,S,D)`：一批有 B 条序列，每条有 N 个注意力头，每个头有 S 个 token 位置，每个位置是 D 维向量。**

| 字母 | 含义 | 回答的问题 |
| --- | --- | --- |
| B | Batch size，批大小 | 有几条序列？ |
| N | Number of heads，注意力头数 | 每条序列有几个头？ |
| S | Sequence length，序列长度 | 每个头沿序列有几个位置？ |
| D | Head dimension，头维度 | 每个位置的向量有几个数？ |

这里的 head 可以先理解为一组独立参与注意力计算的特征向量；同一个 token 在不同 head 上各有一个 D 维向量。token 是序列中的处理单位，不一定恰好等于一个汉字或一个单词。

这些字母按注意力接口的上下文解释。例如这里的 N 是头数；其他数据格式中的 N 可能表示 batch。

## 具体例子

设 `B=2、N=8、S=5、D=128`，则 Query 的 shape 为 `(2,8,5,128)`。

```text
Q[b, n, s, d]
  │  │  │  └─ 向量的第 d 个数
  │  │  └──── 第 s 个 token 位置
  │  └─────── 第 n 个注意力头
  └────────── 第 b 条序列
```

下标从 0 开始：`Q[1,2,4,:]` 表示第 2 条序列、第 3 个头、第 5 个 token 的完整 128 维向量；冒号表示取该轴全部元素。

## 与 BSND 的区别

[BSND](BSND.md) 对应 `(B,S,N,D)`。同一个逻辑数满足：

```text
Q_bnsd[b,n,s,d] = Q_bsnd[b,s,n,d]
```

这表示正确转换后的对应关系。转换需要交换 N、S 两个轴，不能只把 shape 的两个数字改掉。

在标准连续存储下，BNSD 将一个 head 的所有 token 向量排在一起；BSND 将一个 token 的所有 head 向量排在一起。轴交换可以先产生带不同 stride 的视图；若后续接口要求连续存储，还需要整理数据。

## 容易混淆的地方

- S 是存储的序列长度。如果实际长度为 3 和 5，补齐到 5 后 S 为 5，但第一条只有 3 个有效位置，需要用接口约定的长度或 mask 表达。
- Q 的头数与 KV 的头数可以不同；Q 的序列长度与 KV 的序列长度也可以不同。读代码时常分别记为 `N1/N2`、`S_q/S_kv`。
- D 是每个头的向量维度。一个 token 的所有头合起来有 `N × D` 个数。

## 在本仓库中的位置

截至 2026-09-10 的本地代码，`QuantBlockSparseAttn` 的 `quant_mode=2`（MXFP8）路径要求 Query 与输出均为 TND，尚未接受 BNSD。依据是 [Host 属性检查](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_check.cpp) 中的 `layoutQStr/layoutOutStr` 检查。

## 关联概念

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [layout](layout.md) | 属于 | 本概念是注意力张量数据布局的一种 | 已确认 |
| [BSND](BSND.md) | 对比 | 使用相同的四个轴，N 与 S 的顺序不同 | 已确认 |
| [TND](TND.md) | 对比 | BNSD 显式保留 B、S；TND 用 T 汇总有效 token。等长时可先转 BSND，再合并 B、S | 已确认 |

## 参考来源

- [昇腾官方：BNSD 维度输入](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendtbapi/ascendtb_01_0279.html)：BNSD 与 BSND 的维度含义；该页面属于另一接口，不代表本算子的支持范围。
- [本地 MXFP8 接口文档](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)：Q/KV 维度与布局限制。
- [本地非连续 Tensor 说明](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/context/non_contiguous_tensor.md)：shape、stride 与 offset。
- [已有布局学习笔记](../07-features/operator-map.md)：连续存储偏移公式与布局泛化的代码入口。
