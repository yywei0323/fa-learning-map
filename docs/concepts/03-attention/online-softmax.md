# FA 的 Softmax、分块更新与 expMax

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

原始问题：一行 attention score 沿 KV 方向拆成多块后，为什么要更新最大值以及 emax，才能保持计算结果不变？

**核心：每块都要维护累计行最大值 $m$、以该最大值为基准的指数和 $l$、未归一化的输出累积 $u$。最大值从旧值变到新值时，历史 $l$ 和历史 $u$ 都要乘 $\exp(m_{old}-m_{new})$；最后用 $u/l$ 得到输出。**

核查日期：2026-09-10；本地 HEAD：`1fe74acf0a60fea0ed143c3cda8054f6c373063f`。实际算子目录为 `attention/flash_attention_score`。本文主要对照 `arch22` 的调用流程，以及 `arch35` 的 `ProcessVec1Nd` / 对齐 128 列的普通浮点分支；后者的底层实现位于 `attention/common/op_kernel/arch35/vf`。

推导先忽略 dropout、sink 和量化，令 $x$ 已包含实际路径的缩放、位置偏置和 mask 效果。不同 `pseType` 的运算顺序见[算子 README](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/flash_attention_score/README.md)。数学等价不表示所有浮点实现逐位相同，量化精度另见[笔记 07](../07-features/quantization.md)。

## 1. Softmax 算的是一行的权重

固定一个 batch、一个 head、一个 Query。该 Query 对所有 Key 的打分构成一行 $x_1,\ldots,x_n$：

$$
p_j=\operatorname{softmax}(x)_j=\frac{e^{x_j}}{\sum_{k=1}^{n}e^{x_k}},
\qquad O=\sum_{j=1}^{n}p_jv_j.
$$

$v_j$ 是第 $j$ 个 Value 向量，因此 $p_j$ 是标量，$O$ 是长度为 $D_v$ 的向量。

直接求 $e^{x_j}$ 可能上溢。令 $m=\max_j x_j$，分子、分母同时乘 $e^{-m}$：

$$
p_j=\frac{e^{x_j-m}}{\sum_k e^{x_k-m}}.
$$

减去同一个数不改变 softmax；选择最大值可以使所有指数输入 $x_j-m\leq 0$，至少一个有效位置的指数为 1。例如 `[1000,1001]` 可改成计算 `[exp(-1),1]`，不必先计算两个巨大的指数。

这里“一行拆分”是沿 **S2 / Skv / KV 序列轴** 切列。仅沿 S1 / Query 轴拆行时，每行的 softmax 仍相互独立，不需要跨不同 Query 合并最大值。

## 2. 为什么不能每块各算 Softmax 后直接拼接

把 `[1,2,3,4]` 分成 `[1,2]` 和 `[3,4]`。两块各自 softmax 都约为 `[0.268941,0.731059]`；拼起来总和为 2，而且没有保留第二块整体分数更高这件事。

原因是每块使用了自己的分母。即使让拼接结果再除以 2，也只是给两块相同总权重，仍然不等于整行 softmax。

FA 因此先计算**尚未除以整行分母的指数权重**，将分母和输出累积一起在线更新。仓库[设计介绍第 1 节](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/flash_attention_score/docs/FA%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md)第 11–18 行明确写了 `exp[i]`、历史 PV 的乘加和最后除以 sum。

## 3. 每行需要保存的状态与更新公式

把一行按列分成 $B_1,B_2,\ldots,B_T$，本文块编号从 1 开始，源码 `s2LoopCount` 从 0 开始。

| 符号 | 含义 | 每行大小 |
| --- | --- | --- |
| $r_i=\max_{j\in B_i}x_j$ | 当前块自己的最大值 | 1 个标量 |
| $m_i$ | 已处理所有块的累计最大值 | 1 个标量 |
| $l_i$ | 已处理所有元素在 $m_i$ 基准下的指数和 | 1 个标量 |
| $u_i$ | 已处理所有元素在 $m_i$ 基准下的加权 Value 和，未归一化 | $D_v$ 个数 |
| $\alpha_i$ | 本次历史结果的修正系数 | 1 个标量 |

已有状态是 $m_{i-1},l_{i-1},u_{i-1}$。处理下一块时：

$$
\boxed{m_i=\max(m_{i-1},r_i)}
$$

$$
\boxed{\alpha_i=e^{m_{i-1}-m_i}}
$$

当前块直接以**新的累计最大值**为基准计算：

$$
\widetilde p_{ij}=e^{x_j-m_i},\quad j\in B_i,
\qquad b_i=\sum_{j\in B_i}\widetilde p_{ij},
\qquad t_i=\sum_{j\in B_i}\widetilde p_{ij}v_j.
$$

$$
\boxed{l_i=\alpha_i l_{i-1}+b_i},
\qquad
\boxed{u_i=\alpha_i u_{i-1}+t_i}.
$$

处理完所有块后：

$$
\boxed{O=\frac{u_T}{l_T}}.
$$

第一块有效时，直接初始化 $m_1=r_1,l_1=b_1,u_1=t_1$。统一数学写法可用 $m_0=-\infty,l_0=0,u_0=0$，此时第一块 $\alpha_1=0$；真实实现一般用 `NoUpdate` 分支避免读取未初始化的历史 buffer。

### 3.1 emax 到底是什么

你提到的 emax，若指代码中保存到 `softmaxExpBuf` / `expMaxTensor` 的量，就是 $\alpha_i=e^{m_{i-1}-m_i}$。它既不是 $m_i$ 本身，也不是单独的 $e^{m_i}$。

它是“旧指数基准换成新指数基准”的比例：

$$
e^{x-m_i}=e^{x-m_{i-1}}\,e^{m_{i-1}-m_i}.
$$

因为 $m_i\geq m_{i-1}$，有效历史状态下 $0<\alpha_i\leq 1$；浮点下可能下溢为 0。新块没有更大的最大值时，$\alpha_i=1$，历史结果不用缩小，但仍要加上当前块的贡献。**每次维护的是状态，最大值本身不一定每次改变。**

### 3.2 为什么历史输出也要乘这个系数

同一行所有历史元素都乘相同的 $\alpha_i$，因此可以将它提到求和外面：

$$
\sum_{j\in B_{<i}}e^{x_j-m_i}v_j
=\alpha_i\sum_{j\in B_{<i}}e^{x_j-m_{i-1}}v_j
=\alpha_i u_{i-1}.
$$

于是无需取回旧的全部 score / 指数权重，也无需重新算旧块的 PV，只须修正长度为 $D_v$ 的历史输出累积。

历史分母同理，所以不能只更新 max，也不能只修正 sum。分子、分母必须处于同一指数基准。

## 4. 分块为什么保持数学结果不变

每轮结束都保持以下不变量：

$$
m_i=\max_{j\in B_1\cup\cdots\cup B_i}x_j,
$$

$$
l_i=\sum_{j\in B_1\cup\cdots\cup B_i}e^{x_j-m_i},
\qquad
u_i=\sum_{j\in B_1\cup\cdots\cup B_i}e^{x_j-m_i}v_j.
$$

第一块显然成立；下一块通过上面的指数换基恒等式修正历史，再加上新块，仍然成立。最终：

$$
\frac{u_T}{l_T}
=\frac{\sum_j e^{x_j-m_T}v_j}{\sum_j e^{x_j-m_T}}
=\sum_j\frac{e^{x_j}}{\sum_k e^{x_k}}v_j.
$$

这给出分块与整行计算的等价证明。它要求处理的是相同的有效元素集合以及相同的 score / Value；删掉有效块或改变 mask 会改变数学问题。

若确实需要完整概率矩阵，已输出的旧块 $\widetilde p$ 还要换到最终基准并除以 $l_T$，不能原样拼接。FA 计算 $O$ 时通过累积 $u$ 避免保存完整概率矩阵。

## 5. 对照源码：从调用到实际运算

### 5.1 arch22：看清第一块、更新块和最后一块

文件：[flash_attention_score_s1s2_bn2gs1.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h)。

`SoftMaxCompute` 第 1927–1951 行取出三个行统计量 buffer：

| 上层变量 | 对应数学量 |
| --- | --- |
| `maxUb` / `softmaxMaxBuf` | 累计行最大值 $m_i$ |
| `sumUb` / `softmaxSumBuf` | 累计指数和 $l_i$ |
| `expUb` / `softmaxExpBuf` | 历史修正系数 $\alpha_i$ |

第 1954 行判断 `extraInfo.s2LoopCount == 0 && !hasSink`。第一块调用 `SoftmaxFlashV2<T, false, ...>`；后续调用 `SoftmaxFlashV2<T, true, ...>`。这里表示初始化与读取历史状态的两种路径。API 的函数体不在本文件中，下面用 arch35 的显式实现展开对应数学操作。

输出侧第 2233–2244 行展示了关键流程：第一块直接保存当前 MM2 结果；后续先调用 `Bmm2ResultMul` 缩放历史结果，再 `Add` 当前 MM2 结果；最后一块才调用 `Bmm2ResultDiv`。

- `Bmm2ResultMul`：第 2270 行读取 `softmaxExpBuf`，第 2287 / 2296 行等执行按行广播的 `Mul`，对应 $\alpha_i u_{i-1}$。
- `Bmm2ResultDiv`：第 2323–2324 行读取 `softmaxSumBuf`，第 2334 / 2344 行等执行按行 `Div`，对应 $u_T/l_T$。

### 5.2 arch35：当前块最大值、指数与局部指数和

调用入口：[flash_attention_noquant_block_vec_base.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h) 中 `ProcessVec1Nd`。

第 1048–1050 行的 `sumUb`、`maxUb`、`expUb` 与上表同义。无 sink 时，第 1176–1183 行调用 `ProcessVec1Vf<..., false, ...>` 处理第一块，第 1210–1216 行调用 `ProcessVec1Vf<..., true, ...>` 处理后续块。

沿着 [vf_mul_sel_softmaxflashv2_cast_nz.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h) 第 375–386 行的 `isUpdate` 分支，进入 [vf_basic_block_aligned128_update.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/common/op_kernel/arch35/vf/vf_basic_block_aligned128_update.h) 的 `ProcessVec1UpdateImpl128VF`。

该函数先在第 179–188 行对当前块做 `MAX` 归约，得到 $r_i$，然后在第 196–200 行读取历史 max 并合并：

```cpp
LoadAlign(vreg_in_max, inMaxUb);
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
LoadAlign(vreg_input_max, tmpMaxUb2);
Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);
StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
```

这里 `vreg_input_max` 对应 $r_i$，`vreg_in_max` 对应 $m_{i-1}$，`vreg_max_new` 对应 $m_i$。

第 213–227 行广播 $m_i$ 后计算当前块指数与指数和，核心语句摘录如下：

```cpp
ExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all);
ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max, preg_all);
Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum, preg_all);
```

`ExpSub(dst,a,b,mask)` 在这里对应 $\exp(a-b)$；`even` / `odd` 是寄存器中拆开的数据部分，合起来才是一行的归约。当前块指数和写入临时 `tmpExpSumUb`，它对应 $b_i$，还不是累计 $l_i$。

**变量名要结合所在函数阅读：**这个底层 VF 的参数 `expUb` 指向 `dstTensor`（第 376 行），保存的是当前块指数权重；修正系数的参数叫 `expMaxUb`。不要将它与上层 `ProcessVec1Nd` 的 `expUb` 混为一谈。VF 参数 `m` 则是处理行数，也不是本文的数学最大值 $m_i$。

### 5.3 arch35：expMax 与累计 sum 的真正更新位置

`ProcessVec1Nd` 第 1312–1314 行在非首块调用：

```cpp
UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
```

进入 [vf_mul_sel_softmaxflashv2_cast_nz.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h) 的 `UpdateExpSumAndExpMaxImplVF`，普通 `useNz == false` 分支第 194–205 行：

```cpp
LoadAlign(vreg_max, tmpMaxUb);       // m_i
LoadAlign(vreg_in_max, inMaxUb);     // m_(i-1)
ExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all); // alpha_i
StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max, preg_all);
LoadAlign(vreg_in_exp_sum, inExpSumUb);  // l_(i-1)
LoadAlign(vreg_exp_sum_brc, tmpExpSumUb); // b_i
Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_sum_update, preg_all);
```

上面是原语句摘录并添加数学注释；对应 $\alpha_i=e^{m_{i-1}-m_i}$ 与 $l_i=\alpha_i l_{i-1}+b_i$。输入、输出允许复用同一 buffer，所以代码先读取旧 max，再写回新 max。

### 5.4 arch35：历史输出更新与最终归一化

文件：[vf_flashupdate_new.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/common/op_kernel/arch35/vf/vf_flashupdate_new.h)。普通浮点分支第 55、63 行：

```cpp
Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
```

`vreg_input_pre` 是 $u_{i-1}$，`vreg_input_cur` 是当前 MM2 得到的 $t_i$，所以两句就是 $u_i=\alpha_i u_{i-1}+t_i$。

末轮函数第 243–244 行再执行：

```cpp
Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
```

上层 `flash_attention_noquant_block_vec_base.h` 第 1421–1453 行选择 `FlashUpdateNew` 或 `FlashUpdateLastNew`；若整行只有一块，第 1458–1462 行调用 `LastDivNew` 直接做最后的除法。

以上解释针对普通浮点分支；文件中 FP8 / MLA 全量化分支还会乘反量化 scale，不能仅复制这两句概括所有模板。

## 6. 两块算例：逐步看到历史贡献缩小

取一行 $x=[1,2\mid3,4]$，对应的标量 Value 为 $v=[10,20\mid30,40]$。

| 块 | 当前块 $r_i$ | 累计 $m_i$ | $\alpha_i$ | 累计 $l_i$ | 累计 $u_i$ |
| --- | --- | --- | --- | --- | --- |
| `[1,2]` | 2 | 2 | 首块无需历史修正 | 1.367879441 | 23.678794412 |
| `[3,4]` | 4 | 4 | $e^{-2}\approx0.135335283$ | 1.553001793 | 54.240959584 |

第一块：

$$
l_1=e^{-1}+1,\qquad u_1=10e^{-1}+20.
$$

第二块的新最大值是 4，当前块的 $b_2=e^{-1}+1$，$t_2=30e^{-1}+40$。修正历史后：

$$
l_2=e^{-2}(e^{-1}+1)+(e^{-1}+1)
=e^{-3}+e^{-2}+e^{-1}+1,
$$

$$
u_2=e^{-2}(10e^{-1}+20)+(30e^{-1}+40)
=10e^{-3}+20e^{-2}+30e^{-1}+40.
$$

最终 $O=u_2/l_2\approx34.926527346$，整行 softmax 权重约为 `[0.032058603, 0.087144319, 0.236882818, 0.643914260]`，直接与 Value 加权也得到同一个结果。

如果分母更新正确，却忘记给历史 $u_1$ 乘 $e^{-2}$，输出会变成约 **48.110168317**。所有 Value 都在 `[10,40]` 内，合法 softmax 加权平均不应超过 40，这也是此例很直观的排错线索。

## 7. 阅读其他算法写法时的两个区别

**若保存的是已归一化输出 $O_i$：**因为 $u_{i-1}=l_{i-1}O_{i-1}$，更新要写成：

$$
O_i=\frac{\alpha_i l_{i-1}O_{i-1}+t_i}{l_i}.
$$

本笔记追踪的代码保存未归一化 $u$，最后才除法，不能把两种公式混用。

**若两个块已经各自独立算完统计量：**设 A、B 各自在自己的最大值基准下保存 $(m_A,l_A,u_A)$ 与 $(m_B,l_B,u_B)$，合并公式为：

$$
m=\max(m_A,m_B),\qquad
l=e^{m_A-m}l_A+e^{m_B-m}l_B,
$$

$$
u=e^{m_A-m}u_A+e^{m_B-m}u_B.
$$

这时两边都可能需要换基。前面的顺序实现已经用新 $m_i$ 计算当前块，因此当前块无须再额外乘一次 $e^{r_i-m_i}$。

## 8. Mask、精度与验证范围

- **全被 mask 的行：**数学上没有可归一化的有效元素，不能直接代入产生 $-\infty-(-\infty)$。所追踪 arch35 调用者第 1318–1321 行有按模式启用的 `InvalidLineProcess`；arch22 也有 `AdjustSoftMaxRes`。应查看所选模板和接口约定，不能把普通行公式直接套到无效行。
- **Dropout：**若保留概率为 $q$、固定 mask 为 $d_j\in\{0,1\}$，分子贡献改成 $\widetilde p_{ij}(d_j/q)v_j$，分母仍统计 dropout 前的指数和。128 列实现先在第 223–227 行统计 sum，再在第 242–255 行处理 dropout，顺序与此一致。
- **Sink：**会影响 softmax 的统计量及首轮处理。arch22 第 1936–1944 行存在用 sink 初始化 max/sum 的分支；本文普通首块公式不覆盖该扩展。
- **浮点：**分块会改变求和顺序；指数近似、累加舍入以及转成 BF16 / FP8 等都会产生误差。`expMax` 修正解决分块基准一致性，不能恢复量化已经丢失的信息。

附带可运行的 [CPU 公式验证脚本](../../../examples/concepts/verify_online_softmax.py)，从仓库根目录运行：

```powershell
python examples/concepts/verify_online_softmax.py
```

已用 Python 双精度对整行参考式和在线式做了 **526 种分块验证**，包括单元素块、尾块、最大值不变、重复分数、极端分数和部分 mask；最大输出绝对差为 `6.217e-15`，两块算例与上表一致。脚本对全 mask 块采用跳过，对全 mask 行显式报错，这是教学约定。

这项验证检查本文数学模型，没有执行 Ascend NPU Kernel，也不是 FP8 / BF16 硬件精度或性能测试。
