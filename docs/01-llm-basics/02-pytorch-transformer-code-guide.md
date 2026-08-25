# 第 1 章补充｜掩码自注意力与多头注意力

> 对照阅读：[Happy-LLM 第二章 2.1.5～2.1.6](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84.md)  
> 学习方式：先理解“每一行在看谁”，再运行 [配套 Notebook](../../examples/ch01/pytorch_transformer_shapes.ipynb)。代码涉及的 PyTorch 操作直接在代码注释里解释，不单独拆成语法课。

---

## 1. 先把三个名字分开

“掩码自注意力”和“多头注意力”听起来像两种完全不同的算法，其实它们描述的是不同问题。

| 名称 | 回答的问题 | 一句话理解 |
| --- | --- | --- |
| 自注意力 Self-Attention | Q、K、V 从哪里来？ | 都来自同一段序列，但分别经过不同投影 |
| 掩码/因果 Causal Mask | 当前 token 可以看哪些位置？ | 只能看自己和过去，不能看未来 |
| 多头 Multi-Head | 同一段序列从几个子空间建模？ | 多组 Q/K/V 并行计算，再合并结果 |

因此，主流 Decoder-only 大模型里最常见的是：

> Causal Multi-Head Self-Attention：带因果 Mask 的多头自注意力。

它同时满足：

1. Q、K、V 来自同一段输入；
2. 每个位置不能看未来；
3. 有多个注意力头并行计算。

---

## 2. 掩码自注意力到底“遮住”了什么？

先不要想多头，只考虑单头和 4 个 token。

假设训练序列是：

| 位置 | 输入 token | 该位置用于预测 |
| --- | --- | --- |
| 0 | BOS | 我 |
| 1 | 我 | 喜欢 |
| 2 | 喜欢 | 苹果 |
| 3 | 苹果 | EOS |

自注意力会产生一个 4×4 的分数矩阵：

- 行：Query，表示“当前是哪个位置在查询”；
- 列：Key，表示“当前 Query 正在查看哪个位置”。

例如第 2 行第 0 列表示：“喜欢”这个位置，对 BOS 位置应该分配多少注意力。

如果没有 Mask，第 0 行也能看到“我、喜欢、苹果”，这等于预测“我”时偷看了答案和未来内容。

### 2.1 可见范围

| Query \ Key | BOS 0 | 我 1 | 喜欢 2 | 苹果 3 |
| --- | ---: | ---: | ---: | ---: |
| BOS 0 | 可见 | 遮住 | 遮住 | 遮住 |
| 我 1 | 可见 | 可见 | 遮住 | 遮住 |
| 喜欢 2 | 可见 | 可见 | 可见 | 遮住 |
| 苹果 3 | 可见 | 可见 | 可见 | 可见 |

对应的数值 Mask 是：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

主对角线保留，是因为位置 i 可以看到自己；主对角线上方是未来位置，所以填负无穷。

### 2.2 Mask 不是把 token 删除

Attention 原始分数为：

$$
Score=\frac{QK^T}{\sqrt D}
$$

先加 Mask，再做 Softmax：

$$
P=Softmax(Score+M)
$$

因为 $e^{-\infty}=0$，所以被遮住的位置在 Softmax 后权重为 0。它们仍然位于矩阵中，只是不再参与 Value 的加权求和。

### 2.3 用一段代码看清 Mask

下面不先讲 full、triu、dim 等语法；每个操作直接写在它产生作用的位置。

~~~python
import torch
import torch.nn.functional as F

tokens = ["BOS", "我", "喜欢", "苹果"]
S = len(tokens)

# 假设 QK^T / sqrt(D) 已经算完。
# 行是 Query 位置，列是 Key 位置，所以 Shape 是 [S, S]。
scores = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
])

# 先创建一个全部为 -inf 的 [S, S] 矩阵。
mask = torch.full((S, S), float("-inf"))

# diagonal=1 表示只保留主对角线上方。
# 主对角线和下方会变成 0，正好表示“自己和过去可见”。
mask = torch.triu(mask, diagonal=1)

# 被遮位置：有限分数 + (-inf) = -inf。
masked_scores = scores + mask

# dim=-1 表示沿每一行的 Key 位置做 Softmax。
# 每个 Query 都得到一组“应该关注哪些 Key”的概率。
weights = F.softmax(masked_scores, dim=-1)

print("Mask:")
print(mask)

print("Softmax 后的注意力权重:")
print(weights)
~~~

运行后应观察到：

- 第 0 行只能在第 0 列得到 1；
- 第 1 行只有前两列可能非零；
- 第 2 行只有前三列可能非零；
- 最后一行可以关注全部位置；
- 每一行的权重之和都是 1。

### 2.4 有 Mask 为什么还能并行训练？

“不能看未来”不等于“必须逐 token 串行计算”。

训练时，QKᵀ 的 4 行仍然一次性计算完成，Mask 也一次性加到整个矩阵上。四个位置的预测仍可并行产生，只是每一行被规定了不同的可见范围。

~~~mermaid
flowchart TD
    A["一次计算全部 QKᵀ"] --> B["得到 S × S 分数矩阵"]
    B --> C["一次加上因果 Mask"]
    C --> D["每行独立 Softmax"]
    D --> E["并行得到所有位置输出"]
~~~

真正自回归推理时，下一个 token 尚未产生，所以 Decode 仍然是逐 token 的；这与训练中的矩阵并行不是一回事。

---

## 3. 多头注意力到底“多”在哪里？

### 3.1 它不是把句子分成多段

多头注意力不会把 BOS、我、喜欢、苹果分给不同头。每一个头都看到完整序列。被拆分的是每个 token 的隐藏向量。

假设：

- B=1：一条序列；
- S=4：四个 token；
- H=8：每个 token 有 8 维隐藏向量；
- N=2：两个注意力头；
- D=H/N=4：每个头 4 维。

输入 Shape 是 $X:[1,4,8]$。

Q/K/V 投影后仍然是 [1,4,8]，然后把最后的 8 维拆成 2 个头、每头 4 维：

$$
[1,4,8]\rightarrow[1,4,2,4]
$$

为了让每个头都能执行自己的矩阵乘法，再交换 Head 与 Sequence 维：

$$
[1,4,2,4]\rightarrow[1,2,4,4]
$$

此时含义是 [B,N,S,D]：1 个 batch，2 个 head，每头都有完整的 4 个 token，每个 token 使用 4 维表示。

### 3.2 每个头为什么能学到不同关系？

三个投影层的参数可以按输出维看作多块独立参数：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

拆成 N 个头后，第 i 个头实际使用自己对应的参数区域：

$$
head_i=Attention(Q_i,K_i,V_i)
$$

不同头初始化不同、训练梯度不同，因此可能学习不同关系，例如相邻 token、主谓关系或远距离依赖。这些只是便于理解的可能结果，不表示开发者预先规定了每个头的职责。

---

## 4. 多头掩码自注意力：直接在代码中理解

下面只保留核心流程。每段注释都同时回答“算什么、Shape 怎么变、为什么这样变”。

~~~python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        # H 必须能平均分给 N 个头，否则每头维度 D 不是整数。
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size       # H，例如 8
        self.num_heads = num_heads           # N，例如 2
        self.head_dim = hidden_size // num_heads  # D=H/N，例如 4

        # 同一个输入 x 分别经过三套参数，得到 Q、K、V。
        # Linear 只改变最后一维；这里 H->H，所以 Shape 仍是 [B,S,H]。
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # 多头结果拼接后还是 [B,S,H]，再用 W_o 混合各头信息。
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        # 例如 B=1、S=4、H=8，则 x.shape=[1,4,8]。
        B, S, H = x.shape

        # “自注意力”指来源都是 x，不代表 Q、K、V 数值相同。
        q = self.q_proj(x)  # [B,S,H]
        k = self.k_proj(x)  # [B,S,H]
        v = self.v_proj(x)  # [B,S,H]

        # 把 H 拆成 N×D。
        # view 只重解释最后 8 个数为“2 个头，每头 4 个数”。
        q = q.view(B, S, self.num_heads, self.head_dim)  # [B,S,N,D]
        k = k.view(B, S, self.num_heads, self.head_dim)  # [B,S,N,D]
        v = v.view(B, S, self.num_heads, self.head_dim)  # [B,S,N,D]

        # 把 Head 维放到 Sequence 维之前。
        # 这样 B×N 组 [S,D] 矩阵可以并行计算。
        q = q.transpose(1, 2)  # [B,N,S,D]
        k = k.transpose(1, 2)  # [B,N,S,D]
        v = v.transpose(1, 2)  # [B,N,S,D]

        # 每个头分别计算每个 Query 对每个 Key 的相似度。
        # [B,N,S,D] @ [B,N,D,S] -> [B,N,S,S]
        # 最后两个 S 分别是 Query 位置和 Key 位置。
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # 构造 [S,S] 因果 Mask。
        # 主对角线上方是未来位置，值为 -inf；其他位置为 0。
        mask = torch.full(
            (S, S),
            float("-inf"),
            device=x.device,
        )
        mask = torch.triu(mask, diagonal=1)

        # scores 是 [B,N,S,S]，mask 是 [S,S]。
        # PyTorch 会把同一份 mask 广播给所有 batch 和 head。
        scores = scores + mask

        # 每个 Query 沿所有 Key 位置归一化。
        # Shape 仍为 [B,N,S,S]；每一行之和为 1。
        # 临时转 FP32 是为了让指数与求和更稳定。
        weights = F.softmax(scores.float(), dim=-1).type_as(q)

        # 用注意力概率对 V 加权求和。
        # [B,N,S,S] @ [B,N,S,D] -> [B,N,S,D]
        context = torch.matmul(weights, v)

        # 把多头结果放回 token 维之后：[B,S,N,D]。
        context = context.transpose(1, 2)

        # transpose 通常得到非连续内存视图。
        # contiguous 整理后，再把 N×D 合回 H。
        context = context.contiguous().view(B, S, H)  # [B,S,H]

        # 输出投影混合不同头的信息。
        output = self.out_proj(context)  # [B,S,H]

        # 返回 weights 只是为了学习时观察。
        return output, weights
~~~

### 4.1 用具体参数运行

~~~python
torch.manual_seed(7)

B, S, H, N = 1, 4, 8, 2

# 模拟 4 个 token 的隐藏状态。
x = torch.randn(B, S, H)

attention = CausalMultiHeadSelfAttention(
    hidden_size=H,
    num_heads=N,
)

output, weights = attention(x)

print("输入:", x.shape)          # [1,4,8]
print("输出:", output.shape)     # [1,4,8]
print("权重:", weights.shape)    # [1,2,4,4]
print("第 0 个头:")
print(weights[0, 0])
print("第 1 个头:")
print(weights[0, 1])
~~~

你应重点观察：

1. 两个头各有一张 4×4 注意力矩阵；
2. 两张矩阵主对角线上方都为 0；
3. 两张矩阵的具体权重通常不同；
4. 输入输出都是 [B,S,H]，所以可以做残差连接。

---

## 5. 把整个 Shape 流串起来

~~~mermaid
flowchart TD
    A["x: B × S × H"] --> B["Q/K/V 投影: B × S × H"]
    B --> C["拆头: B × S × N × D"]
    C --> D["转置: B × N × S × D"]
    D --> E["QKᵀ: B × N × S × S"]
    E --> F["加 Mask + Softmax"]
    F --> G["乘 V: B × N × S × D"]
    G --> H["合头 + Wₒ: B × S × H"]
~~~

用示例数字代入：

| 步骤 | Shape |
| --- | --- |
| 输入 | [1,4,8] |
| Q/K/V 投影 | 各自 [1,4,8] |
| 拆成 2 个头 | 各自 [1,4,2,4] |
| 调整维度 | 各自 [1,2,4,4] |
| 每头 QKᵀ | [1,2,4,4] |
| 加 Mask、Softmax | [1,2,4,4] |
| 乘 V | [1,2,4,4] |
| 合并两个头 | [1,4,8] |
| 输出投影 | [1,4,8] |

示例里恰好 S=4、D=4，所以多个 Shape 都出现数字 4，容易混淆。必须根据维度语义区分：

- Score 的 [1,2,4,4] 是 [B,N,S_q,S_kv]；
- Context 的 [1,2,4,4] 是 [B,N,S,D]；
- 数字相同不表示含义相同。

运行 Notebook 练习时，可以把 S 改成 5：

- Score：[1,2,5,5]；
- Context：[1,2,5,4]。

---

## 6. Happy-LLM 原代码逐段对应

| Happy-LLM 代码 | 现在应该怎样读 |
| --- | --- |
| wq/wk/wv 三个 Linear | 从同一输入学习三种角色，因此是 Self-Attention |
| view(B,S,N,D) | 将隐藏维 H 拆成多头 N×D |
| transpose(1,2) | 变成 [B,N,S,D]，让每个头独立做矩阵乘 |
| q @ k.transpose(...) | 得到 [B,N,S_q,S_kv] 的分数矩阵 |
| scores + mask | 将未来位置变成负无穷 |
| softmax(dim=-1) | 每个 Query 对全部 Key 得到权重 |
| weights @ v | 汇聚当前 Query 需要的 Value 信息 |
| transpose + contiguous + view | 将 N 个头重新拼回 H |
| wo | 混合多个头的输出 |

### 6.1 原文代码中需要注意的简化

原文先从 q 取得 seqlen，随后用同一个 seqlen reshape q、k、v。这对普通等长自注意力成立，但通用场景应区分 S_q 与 S_kv。

尤其在 Decode + KV Cache 中：

$$
S_q=1,\quad S_{kv}=历史长度+1
$$

此时：

$$
Q:[B,N,1,D]
$$

$$
K,V:[B,N,S_{kv},D]
$$

$$
QK^T:[B,N,1,S_{kv}]
$$

因此不要形成“Attention Score 永远是 S×S 方阵”的印象。

---

## 7. Prefill 与 Decode 中的 Mask

### Prefill

一次输入整段 Prompt：

- Q、K、V 长度都等于 Prompt 长度；
- 必须使用因果 Mask，防止较早位置看到后续 token；
- Score 通常为 [B,N,S,S]。

### Decode

每次只生成一个新 token：

- 当前 Q 长度通常是 1；
- K/V 包含当前与全部历史 Cache；
- Score 为 [B,N,1,S_kv]。

简单单序列 Decode 中，往往不再需要 Prefill 那种完整 S×S 上三角 Mask；实际系统还可能处理 padding、变长序列、滑动窗口等其他 Mask。

这也是 FIA 推理算子必须同时处理 Prefill 与 Decode 不同 Shape 的原因。

---

## 8. 它和 FA/FIA 的关系

普通 PyTorch 实现会依次产生：

$$
QK^T
\rightarrow Score
\rightarrow MaskedScore
\rightarrow SoftmaxProbability
\rightarrow PV
$$

如果完整 Score/Probability 都写入外部显存，长序列会产生大量读写。

FA/FIA 关注的是：

1. 将 Q/K/V 分块搬入片上存储；
2. 分块计算 QKᵀ；
3. 在线完成 Mask、Scale 和 Softmax；
4. 立即与 V 相乘；
5. 尽量避免完整 S_q×S_kv 中间矩阵反复落到外存。

因此现在需要真正掌握的是：

- Score 的行是 Query，列是 Key；
- Mask 改变可见范围；
- Softmax 沿 Key 维计算；
- 多头把 H 拆成 N×D；
- Prefill 与 Decode 的 S_q/S_kv 不同。

---

## 9. 建议学习顺序与自检

1. 手画 4×4 Mask，并逐行说出谁能看谁；
2. 运行 Notebook 的单头 Mask 实验，检查上三角权重是否为 0；
3. 运行多头代码，观察两个头的权重矩阵是否不同；
4. 把 S 从 4 改成 5，区分 Score 的 S×S 和 Context 的 S×D；
5. 最后把 S_q 改成 1、S_kv 改成 6，观察 Decode Shape。

自检问题：

1. Mask 为什么加在 Softmax 之前？
2. 多头是拆 token，还是拆每个 token 的隐藏维？
3. 为什么不同头可能学到不同关系？
4. Score 的最后一维为什么是 Key 位置？
5. Decode 时为什么 Score 不一定是方阵？

如果这五问都能独立回答，再继续学习 KV Cache、GQA 与 FIA。

