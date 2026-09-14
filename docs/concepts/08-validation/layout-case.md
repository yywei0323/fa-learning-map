# 综合案例：从 TND 扩展到 BNSD / BSND

[概念目录](../README.md) · [布局词条](../04-layout/layout.md)

本页将已有设计笔记整理为概念案例。它说明不同层为什么必须同步修改，不表示此次对算子代码进行了改动或确认 PR 已合入。

## 固定版本，区分三类材料

| 材料 | 版本 / 状态 | 能说明什么 |
|---|---|---|
| 原始学习基线 | 1fe74acf0a60fea0ed143c3cda8054f6c373063f | 该快照的 QBSA-MXFP8 Q/O 约束为 TND，KV 为 PA_BNBD |
| 原设计笔记 | 2026-09-10 的实现建议 | 提出固定 S、布局寻址、scale 同步换轴等方案 |
| PR #11704 快照 | head ee2a4c58086472d1297ae2f4a7174c9561b7bfe8；diff base 125200e0afd4b13719694a9014e6053b89e1ad09 | 这份具体扩展实现的接口与代码；不是最新合入状态 |

[原 PR](https://gitcode.com/cann/ops-transformer/pull/11704)；本页依据本地保留的固定快照及对照笔记，不重复宣称今天的 PR 状态。

## 这次扩展改变了什么

对固定且全有效的 Query S，四维 shape 本身携带 B、S、N、D，Host 可以解析 S 后下传。TND 的总 T 不携带各样本边界，通常需要长度前缀表。

**“Query S 不查表”仅影响 Query 长度来源。KV 的实际长度、分页映射、稀疏索引仍可能需要读取。** 四维 shape 若含 padding，其存储 S 也不自动等于有效 S。

## 每层需要一致

| 位置 | 必须跟随布局改变的事项 |
|---|---|
| Torch eager / Meta 输出构造 | 输入维数、输出 shape、None 参数约定 |
| Host parser / check | B/S/N/D 的轴解释、合法组合、整数范围和长度来源 |
| InferShape / InferDataType | attention_out 与 LSE 的形状、类型 |
| TilingData | 固定 qSeqSize、Query/scale 行跨度等执行参数 |
| TilingKey / 模板声明 | 真正存在新 layout 对应实例 |
| Kernel 地址和搬运 | 起始位置、行跨度、尾块与写回 |
| Empty block 清零 | 空稀疏行也使用正确的 Q/O/LSE 索引 |
| Metadata | 所描述的 Query 任务不得越过实际 S |
| Golden / tests | 用相同逻辑值构造三种布局，转换回同一语义后比较 |

仅修改 shape 或 layout 字符串并不移动数据，不能完成扩展。

## 最容易遗漏的 LSE 约定

PR 快照中，BNSD 与 BSND 的 attention_out 跟随 Q，**两者的 LSE 都使用 [B,N,S]**。LSE 去掉 D，但没有“必须保留 Q 剩余轴顺序”的通用规则。

| 布局 | Q / O 的 token-head 槽位 | PR 的 LSE 槽位 |
|---|---|---|
| BNSD | (b*N+n)*S+s | (b*N+n)*S+s |
| BSND | (b*S+s)*N+n | (b*N+n)*S+s |

Q 地址为 querySlot*D+d；O 地址为 querySlot*Dv+dv。当前 D=128 的 q_descale 每槽 C=D/32=4 个 scale，地址为 querySlot*C+c。所有偏移先按元素解释，再结合 dtype 字节数。

在本 PR 的 Torch 接口中，BSND/BNSD 对 Query 长度参数要求 None。零元素 Tensor 仍是有值的 Tensor；测试适配层可能先把空占位转换成 None，不能把两个调用层的行为混为一谈。

## 边界、设计与验证

既有审查笔记指出：当稀疏 Query 块容量大于 ceil(S/blockSize) 时，需要检查多余任务的起点和尾块长度，避免减法下溢或越界。这个问题属于指定快照的待验证边界，不据此断言最新源码仍有缺陷。

合理的布局回归应选择 B>1、N>1、S≠N，让轴交换错误无法被相等维度掩盖；再覆盖尾块、空稀疏行、mask、LSE 开关、GQA、页大小不同于稀疏块大小，以及旧 FP8/TND 路径。

同一逻辑数据跨布局结果一致、Query 长度表确实未读取、性能有所提升，是三个不同验收目标。本次仅整理既有设计和静态证据，没有进行新的 NPU 验收。
