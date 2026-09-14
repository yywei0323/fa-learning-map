# 第 3 层：Attention 数学与 FA

[概念目录](../README.md) · [完整扫盲](../../00-fa-primer.md)

## Q、K、V 和张量形状

Q 是当前查询向量，K 是用于匹配的键向量，V 是按匹配权重汇总的内容向量。名称是角色划分；在 Self-Attention 中它们通常由同一 hidden state 经不同线性投影得到，不是三份原始文本。

统一符号：

| 符号 | 本页含义 |
|---|---|
| B | batch 数 |
| N1 / N2 | Query 头数 / KV 头数 |
| Sq / S1 | Query 序列长度 |
| Skv / S2 | KV 序列长度 |
| D / Dv | Q、K 点积维度 / Value 和每头输出维度 |
| H | 隐藏特征维度；普通拆头时 H=N1*D |
| T | packed 布局中所有样本有效 token 总数，不用于表示 task 数 |

以普通 MHA、BNSD 为例：

$$
Q:[B,N,S_q,D],\quad K:[B,N,S_{kv},D],\quad V:[B,N,S_{kv},D_v]
$$

$$
X=\gamma QK^T+\mathrm{bias}+\mathrm{mask},\qquad
P=\operatorname{softmax}_{S_{kv}}(X),\qquad O=PV
$$

X、P 是 [B,N,Sq,Skv]，O 是 [B,N,Sq,Dv]。转置只交换 K 的最后两个矩阵轴。矩阵的一行固定一个 Query，一列对应一个 Key。

## 公式里的名词

| 概念 | 解释 |
|---|---|
| Score / Logit | Softmax 前的相关分数，不等于最终概率 |
| BMM1 / MM1 | 第一轮矩阵乘，形成 QKᵀ；BMM 表示批量矩阵乘 |
| Scale / softmax_scale | 常用 γ=1/√D，缩放点积分数；与量化 scale 分开辨认 |
| Mask | 排除不可参与的 Query–Key 对 |
| Causal Mask | Query 只能访问因果上可见的 Key；非方形情形需结合绝对位置和接口对齐规则 |
| Padding Mask | 排除为统一存储长度而补齐的无效位置 |
| PSE / 位置偏置 | 对注意力分数引入位置信息；加在缩放之前还是之后由具体语义决定 |
| Softmax | 每行对有效 KV 位置归一化，使权重和为 1 |
| BMM2 / MM2 / PV | 第二轮矩阵乘，以权重汇总 Value |
| Attention Output | 每个 Query/head 的内容向量；不等于词表 logits |

普通有限分数中，被 mask 位置可用 −∞ 表达，其 exp 为 0。**全 mask 行没有普通 Softmax 分母**，必须遵守接口的无效行处理约定。

## MHA、GQA、MQA

| 模式 | 头数关系 | 复用方式 |
|---|---|---|
| MHA | N1=N2 | 各 Q 头有对应 KV 头 |
| GQA | N1=G*N2 | 一组 G 个 Q 头共享一个 KV 头 |
| MQA | N2=1 | 所有 Q 头共享一组 KV |

常见按连续头分组时，`kv_head=q_head//G`；其他头排序必须按接口解释。头数关系不决定采用 BNSD、BSND 还是 TND，它们是不同维度的配置。

## FA、分块与融合

FlashAttention 把 Attention 的运算和数据搬运按块组织，避免完整 S/P 中间矩阵反复写回、读回全局内存。它仍需处理相同的有效 Query–Key 对；普通稠密 FA 并不把二次规模矩阵乘直接变成线性计算。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

| 术语 | 在 FA 中的作用 |
|---|---|
| IO-aware | 把跨存储层的数据读写量纳入算法设计 |
| Tile / Tiling | 用有限的片上容量分批完成大任务 |
| Fusion | 在较少 Kernel 中衔接计算阶段，减少中间写回和启动成本 |
| Online Softmax | 沿 KV 分块时维护累计 max、sum 和输出状态 |
| Recompute | 训练反向中用重算换取更低的中间状态存储 |
| Pipeline / Double Buffer | 让不同块的搬运与计算尽可能交叠 |
| FA2 | 改善非矩阵乘开销、并行划分与线程间通信的算法实现演进；不是 FIA V2 接口 |

FA2 的三个改进方向见[论文](https://arxiv.org/abs/2307.08691)。GPU 论文中的 warp/thread block 不能直接按名字映射成昇腾 AIC/AIV。

“数学等价”以相同输入值、mask、偏置、有效块集合为前提；浮点舍入顺序可能不同。稀疏化改变参与集合，量化改变数值表示，不能把它们的误差归结为普通分块恒等式。

继续读：[Online Softmax 与 expMax 的完整推导](online-softmax.md) · [LSE](LSE.md) · [FA/FIA/QBSA 层级](../07-features/operator-map.md)。
