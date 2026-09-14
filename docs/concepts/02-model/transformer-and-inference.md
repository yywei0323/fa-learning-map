# 第 2 层：Transformer 与推理

[概念目录](../README.md) · [第 1 章完整导读](../../01-llm-basics/README.md) · [第 2 章完整源码课](../../02-transformer-end-to-end/README.md)

## 从文本到模型

| 概念 | 含义 | 与下一层的关系 |
|---|---|---|
| LLM / 语言模型 | 根据已有 token 估计后续 token 的概率 | 为何要重复执行 Decoder |
| Token / Tokenizer | 文本单元及切分、编码工具 | 一个 token 不一定等于一个字或单词 |
| Vocabulary / Token ID | 有限词表和其中的整数编号 | ID 用于查 Embedding |
| Embedding | 把 ID 映射为连续向量 | 输出一般为 [B,S,H] |
| Hidden State / Activation | 某层正在计算的特征向量/中间结果 | Q/K/V 由它投影而来 |
| Parameter / Weight | 训练更新的模型参数 | 与本次激活、跨步 KV Cache 区分 |
| Context length | 一次序列可包含的 token 长度 | 影响 score 矩阵和 KV Cache 规模 |
| Batch / 并发 | 批量处理的样本或请求 | 与序列长度、模型参数量是不同尺度 |

## Transformer 的内部结构

| 概念 | 最小解释 |
|---|---|
| Encoder | 对输入序列建立上下文表示，常用双向 Self-Attention |
| Decoder | 在生成侧使用因果 Self-Attention；Encoder–Decoder 模型中还含 Cross-Attention |
| Decoder-only | 只保留生成主干；现代生成式语言模型的常见结构 |
| Self-Attention | Q/K/V 来自同一序列表示；投影矩阵通常不同 |
| Cross-Attention | Q 来自目标侧，K/V 来自源侧，Sq 可不同于 Skv |
| Q/K/V Projection、Wo | 将 hidden state 投影成 Q/K/V；合头后经输出投影 |
| Multi-head / 拆头合头 | 把隐藏特征 H 分成 N 个 D 维头，H=N*D；不是把 token 分给各个头 |
| Residual / 残差 | 子层输出加回输入，使原始信息和梯度有直接路径 |
| Pre-Norm / Post-Norm | Norm 放在子层之前/残差相加之后；两者代码公式不能混用 |
| LayerNorm | 通常沿每个 token 的特征轴减均值、按方差归一化，再做可学习仿射变换 |
| RMSNorm | 按均方根缩放，通常不减均值；属于路线中已提及、待深入的专题 |
| MLP / FFN | 逐 token 的前馈网络：线性映射、非线性激活、再映射 |
| GELU / SiLU / ReLU | 非线性激活函数；具体模型用哪种要看定义 |
| MoE | 用路由选择部分专家网络；已提及，尚不作为已掌握内容 |
| Positional Encoding | 向模型提供位置信息，如正弦位置编码 |
| RoPE | 对 Q/K 的成对特征按位置旋转，使注意力携带相对位置信息；不等于简单加 Embedding |
| Dropout | 训练时随机屏蔽并缩放部分激活；推理一般关闭 |

Pre-Norm 教学式：`y=x+Attention(Norm(x))`。其中残差保留的是原来的 x，不能误写成归一化后的 x。

## 从输出到训练或生成

| 概念 | 含义 |
|---|---|
| LM Head | 把 hidden state 投影到词表维度 |
| Logits | 未归一化的词表分数，[B,S,Vocab] |
| Softmax 概率 | 将 logits 变成候选 token 的概率；与 Attention 权重的归约对象不同 |
| Greedy / Sampling | 选最大概率 token / 按某种概率策略采样 |
| Temperature / Top-k / Top-p | 控制生成采样分布或候选集合；不是稀疏 KV 选块的同一流程 |
| BOS / EOS / PAD | 起始、结束、补齐用的特殊 token，具体 ID 由 tokenizer 定义 |
| Teacher forcing / 右移标签 | 训练时用真实前缀预测下一个位置；输入与目标错开一位 |
| Cross-entropy / Loss | 用目标 token 的负对数概率构造训练损失 |
| Backpropagation / Optimizer | 按损失求梯度，并据此更新参数 |
| Training / Inference | 学习参数 / 使用参数计算预测；推理不意味着所有计算仅有单 token |

## Prefill、Decode 与 KV Cache

**Prefill 处理当前待预填充的一段输入，产生隐藏状态、各层 K/V 和用于生成的 logits；Decode 随生成步骤追加新 token，并复用历史 K/V。** 分块预填充、复用前缀等实现会让“每次 Prefill 处理整个 Prompt”的说法不再严格成立。

| 对比项 | 基础 Prefill | 普通单 token Decode |
|---|---|---|
| 本次 Query 长度 | 常为 Prompt 长度或当前预填充块长度 | 通常为 1 |
| 可访问 KV | 当前可见上下文 | 历史缓存与本次新增 K/V |
| score 的形状 | [B,N1,Sq,Skv] | [B,N1,1,Skv] |
| 工作特点 | 较多 Query 行，可并行处理 | Query 少，历史 KV 仍可能很长 |

KV Cache 保存每一层的历史 K/V，避免生成下一步时重复计算这部分历史表示。Q 用于本次查询，一般不需要按同样方式跨步保留。KV Cache 仍需占空间并在 Attention 时读取，不能理解成“不再访问历史数据”。[Hugging Face：缓存原理](https://huggingface.co/docs/transformers/cache_explanation)

忽略分页、对齐、额外 scale，K/V head dim 都为 D、元素字节数为 w 时：

$$
\text{KV bytes}=2\times L\times B\times S\times N_{kv}\times D\times w
$$

L 是层数。GQA/MQA 减少 KV 头数；PA 改善分配和寻址；量化改变元素表示及附加 scale；MLA 改变缓存表示。这些方案应分别理解。

Dynamic Cache 随长度增长；Static Cache 预留容量；Offloaded Cache 在不同设备间管理缓存；Quantized Cache 压缩缓存表示。它们只是已见过的策略名称，具体库接口和取舍继续按目标版本学习。

关联：[Q/K/V 和 GQA](../03-attention/attention-and-fa.md) → [布局](../04-layout/layout.md) → [分页与稀疏](../07-features/paging-sparsity.md)。
