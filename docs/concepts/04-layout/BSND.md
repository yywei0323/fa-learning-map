# BSND

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

[返回概念索引](../README.md) · [查看关联关系](../relations.md)

- 学习状态：待理解
- 首次记录 / 最近更新：2026-09-10
- 原始问题：BSND 是什么意思？它和 BNSD、TND 有什么关系？

## 一句话解释

**BSND 是 shape 为 `(B,S,N,D)` 的注意力张量布局：一批有 B 条序列，每条有 S 个 token 位置，每个位置有 N 个注意力头，每个头是 D 维向量。**

B 是批大小，S 是序列长度，N 是注意力头数，D 是每个头的向量维度。详细字母说明见 [BNSD](BNSD.md)。

## 具体例子

设 `B=2、S=5、N=8、D=128`，则 Query 的 shape 为 `(2,5,8,128)`。

```text
Q[b, s, n, d]
  │  │  │  └─ 向量的第 d 个数
  │  │  └──── 第 n 个注意力头
  │  └─────── 第 s 个 token 位置
  └────────── 第 b 条序列
```

下标从 0 开始：`Q[1,4,2,:]` 表示第 2 条序列、第 5 个 token、第 3 个头的完整向量。它与正确转换后的 BNSD 张量中的 `Q_bnsd[1,2,4,:]` 对应。

在标准连续存储下，可以理解为：先放第 1 个 token 的全部 8 个头，再放第 2 个 token 的全部 8 个头，依次继续。

## 与 BNSD、TND 的转换关系

- **BSND 与 BNSD：**交换 S、N 两个轴；需要保持每个元素的逻辑归属，不能只修改 shape 或 layout 字符串。
- **等长且没有补齐位置时，BSND 与 TND：**按 batch 顺序将 B、S 合并成 T，`T=B×S`。例如 `(2,5,8,128)` 可变为 `(10,8,128)`，序列边界为 `[0,5,10]`。逆转换需要知道 B、S。
- **变长且有补齐位置时：**先按真实长度去掉补齐位置，再拼成 TND；反过来恢复普通四维张量时，需要按约定补齐。

例如真实长度为 3 和 5、统一存储长度 S 为 5，直接合并 B、S 会留下 10 个位置，其中有 2 个补齐位置。只保留有效 token 的 TND 应有 `T=8`，并记录边界 `[0,3,8]`。

## 在本仓库中的含义

固定、整段有效的 BSND 输入可以从 `shape[1]` 取得 S，这是按轴定义得出的结论。若包含补齐位置，仅凭 shape 仍不知道各条序列的有效长度。

截至 2026-09-10 的本地 MXFP8 实现，Query 和输出只接受 TND；BSND 尚未成为这条执行路径的可用输入布局。[Host 属性检查](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_check.cpp) 直接校验了这一限制。

## 关联概念

| 关联概念 | 关系类型 | 具体说明 | 确认状态 |
| --- | --- | --- | --- |
| [layout](layout.md) | 属于 | 本概念是注意力张量数据布局的一种 | 已确认 |
| [BNSD](BNSD.md) | 对比 | 都显式保留 B、S，区别在 N 与 S 的轴顺序 | 已确认 |
| [TND](TND.md) | 对比 | 等长且无补齐时可将 BSND 的 B、S 合并；变长时要处理有效长度与补齐位置 | 已确认 |

## 参考来源

- [昇腾官方：BNSD 维度输入](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendtbapi/ascendtb_01_0279.html)：同时给出 BSND 与 BNSD 的轴含义。
- [本地 MXFP8 接口文档](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md)：当前布局限制及长度信息。
- [已有布局学习笔记](../07-features/operator-map.md)：等长转换、变长边界及内存寻址推导。本文转换示例也是依据轴含义作出的推导。
