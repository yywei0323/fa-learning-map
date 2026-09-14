# 分核、计算模板与 Matmul

[概念目录](../README.md) · [Tiling](Tiling.md)

## BNS1 / BN2GS1 分核

BNS1 分核表示沿 batch、Query head、Query 序列块划分可并行任务。对等长、稠密、每个 Query 块独立处理完整可见 KV 的教学模型：

$$
n_{\rm tasks}=B\,N_1\,\lceil S_1/B_r\rceil
$$

Br 是 Query tile 行数。例：B=2、N1=4、S1=1024、Br=128，得到 64 个任务；8 个核可各处理约 8 个任务。任务数不等于核数；本文用 n_tasks，避免与 TND 的总 token 数 T 混用。

GQA 时 N1=N2*G，所以名称也可能写成 BN2GS1。TND 变长场景应按各 batch 的有效 Query 块累加；因果、稀疏等场景中任务的 KV 工作量不同，任务个数平均也不保证耗时平均。

## S1、S2、内外层切块

| 名词 | 在这里的含义 |
|---|---|
| S1 / Query 方向 | 分数矩阵的行；不同 Query 行的 Softmax 相互独立 |
| S2 / KV 方向 | 分数矩阵的列；同一 Query 的多个列块共享归一化状态 |
| Query tile / KV tile | 一次计算的局部 Q 行块和 K/V 列块 |
| Souter / Sinner | 某实现循环的外块/内块命名，必须结合循环和字段定义 |
| Alignment | 地址、长度、格式满足计算与搬运粒度约束 |
| Tail / 尾块 | 不满固定 tile 的最后一块；有效数与分配容量不同 |
| Load balance | 各核实际工作耗时尽量接近 |
| Split-KV / FlashDecode 合并 | 将同一 Query 的 KV 部分分开处理，再按统计量合并结果 |

如果拆 S2 给多个核，不能直接相加各自归一化输出。需要合并 max、sum 与未归一化输出，或使用等价的 LSE 加权公式；见[在线 Softmax 的独立块合并](../03-attention/online-softmax.md)。

## Matmul 的 M / N / K

$$
C_{M\times N}=A_{M\times K}B_{K\times N}
$$

这里 N 是矩阵列数，K 是归约维；它们不一定是 Attention 的 head 数 N 或 Key 张量 K。变量字母相同不代表含义相同。

## 模板参数与运行参数

```cpp
// 仅展示两类参数的区别，不是可直接调用的 Ascend C API。
Matmul<128, 256, 128>(a, b, c, actualM, actualN, actualK);
// 尖括号：编译期配置。
// 普通参数：本次地址、实际尺寸等，可以包含尾块。
```

模板使编译器提前知道类型和部分块配置，可能消除分支、固定地址计算或展开循环；代价包括实例数量和编译体积增加。不是每个输入长度都需要新模板。

## MatmulBase / MatmulFull 的具体含义

以下只对应已核对的 QBSA `common/matmul.h`，不是所有库共享的 API 定义：

| 名称 | 学习快照中的职责 |
|---|---|
| MatmulBase | 根据 singleK/baseK、singleN/baseN 选择 K 分块、N 分块或 Full 路径 |
| MatmulK | 沿归约维分块并累加 |
| MatmulN | 沿结果列方向分块 |
| MatmulFull | 将本次传入块装入相应 L0 操作数空间，再进行矩阵乘 |
| MxMatmulFull | 对 Full 的 MX 包装，带上 scale 和相关类型/装载信息 |

“Full” 指当前调用块的全载路径，不表示把整个模型或整个序列装入片上内存。实际条件顺序见[固定版本源码](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/common/matmul.h#L868)。

## “ND 函数”为什么还需上下文

会议中的 ND 可能指普通 ND 格式相关搬运，如 ND→NZ；也可能是在说计算方向对应的 ND/DN 函数。DN 路径可能形成通常 QKᵀ 的转置方向，后续归约轴也随之变化。只有四个字不能确定含义，应记录完整函数名、文件和参数。见[待核对项](../open-questions.md)。

关联：[layout 与 ND/NZ](../04-layout/layout.md) · [硬件数据流](../05-hardware/ascend.md) · [性能瓶颈](../08-validation/performance-and-tests.md)。
