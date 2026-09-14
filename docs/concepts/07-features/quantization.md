# 量化精度——位宽、scale 与 FA 的数值误差

> 整理于 2026-09-14。文中“当前 / 本地实现”特指学习快照 `1fe74acf…`，不是对上游最新能力的声明。BNSD/BSND 扩展另见[版本对照](../08-validation/layout-case.md)。[目录](../README.md) · [来源与覆盖范围](../sources.md)

原始问题：量化的精度是什么意思？在 FP8 / MXFP8 的 FA 算子里，误差从哪里来？

**量化精度关注：把数值映射到有限的低位宽表示，再还原并参与计算后，与参考结果相差多少。**“8 bit”描述存储格式；它既不表示保留 8 位小数，也不能直接换算成模型准确率下降多少。

核查日期：2026-09-10；本地 HEAD：`1fe74acf0a60fea0ed143c3cda8054f6c373063f`。本文结合[仓库量化介绍](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/context/quant_mode_introduction.md)与 FA 的具体转换代码说明；公式和小例子属于数学推导，没有测量当前 NPU 算子的 FP8 精度。

## 1. 先区分三种“精度”

| 平时的说法 | 实际含义 | 例子 |
| --- | --- | --- |
| 使用什么精度 | 数值格式与位宽 | FP32、FP16、BF16、FP8、INT8 |
| 量化后精度怎样 | 数值与参考结果有多接近 | 最大绝对误差、均方误差、相对误差 |
| 模型精度怎样 | 完整模型的任务表现 | 准确率、困惑度、任务评分 |

低位宽格式不一定在每个数值上都有误差：如果数值恰好可表示，还原就可以完全一致。反过来，输入虽然用 FP8 保存，部分中间计算仍可以用 FP32，不能把整个算子都理解为“只用 8 位计算”。

## 2. 用 INT8 看清舍入与截断

先用最容易理解的对称均匀量化模型。本文选择整数范围 $[-127,127]$，零点为 0；原生 INT8 还可表示 -128，但这个教学量化约定不使用它。设反量化 scale 为 $s>0$：

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s),-127,127\right),
\qquad \widehat x=sq.
$$

$q$ 是低位宽整数，$\widehat x$ 是还原值，误差为 $\delta=\widehat x-x$。仓库有些接口把量化侧乘数命名为 scale；因此读代码时应先确认保存的是 $s$ 还是其倒数 $1/s$。

对 $x=1.23$，选择不同的 $s$：

| $s$ | 可覆盖范围 | 存入的 $q$ | 还原 $\widehat x$ | 绝对误差 |
| --- | --- | --- | --- | --- |
| 0.1 | `[-12.7,12.7]` | 12 | 1.20 | 0.03 |
| 0.01 | `[-1.27,1.27]` | 123 | 1.23 | 0 |
| 0.001 | `[-0.127,0.127]` | 127，发生截断 | 0.127 | 1.103 |

**步长越小，网格越细，但固定整数位宽下覆盖范围也越小。**最后一行说明不能只追求“小 scale”。

如果采用最近舍入，且没有截断、忽略 scale 自身的表示误差，则：

$$
|\widehat x-x|\leq \frac{s}{2}.
$$

这个界只描述均匀量化的舍入误差；发生饱和截断时不再成立。反量化把尺度换回来，却不能恢复舍入或截断丢掉的原始信息。

本例已用[公式验证脚本](../../../examples/concepts/verify_online_softmax.py)核算。表中的零误差按十进制理想模型描述，真实浮点存储本身仍有表示误差。

## 3. FP8 为什么不能照搬 INT8 的固定步长

INT8 加固定 scale 的可表示值是等间距的；FP8 是浮点数，较大数值区域的间距也更大。

对于具有 $t$ 个尾数字段位的正规二进制浮点数：

$$
x=(-1)^{sign}\,2^E\left(1+\frac{M}{2^t}\right).
$$

在同一指数区间 $[2^E,2^{E+1})$ 内，相邻数值间距为：

$$
\operatorname{ULP}=2^{E-t}.
$$

**指数位主要决定可覆盖多大的数量级；尾数位主要决定同一数量级内能分得多细。**下面列的是未乘外部 scale 时，`[1,2)` 区间的相邻数间距，不是任意数值上的固定误差：

| 格式 | 总位数 | 指数位 | 尾数字段位 | `[1,2)` 内间距 |
| --- | --- | --- | --- | --- |
| FP32 | 32 | 8 | 23 | $2^{-23}\approx1.19\times10^{-7}$ |
| FP16 | 16 | 5 | 10 | $2^{-10}\approx0.00097656$ |
| BF16 | 16 | 8 | 7 | $2^{-7}=0.0078125$ |
| FP8 E4M3 | 8 | 4 | 3 | $2^{-3}=0.125$ |
| FP8 E5M2 | 8 | 5 | 2 | $2^{-2}=0.25$ |

FP8 的 E4M3 / E5M2 字段定义见 [FP8 Formats for Deep Learning](https://arxiv.org/html/2209.05433v2)。间距列由上述公式推导。

例如 E4M3 在 `[1,2)` 内可以表示 `1、1.125、1.25、1.375……`。采用最近舍入且 scale 为 1 时，`1.23` 会变成 `1.25`，绝对误差为 `0.02`；到了 `[8,16)`，间距变成 $2^{3-3}=1$。

E5M2 的指数范围更宽，但同一正常数量级内尾数比 E4M3 少一位，网格更粗。FP16 与 BF16 也有类似取舍：BF16 范围更宽，FP16 在共同的正规数范围内分得更细。不能只按“8 位、16 位”判断所有场景中的误差大小。

FP8 量化可概括为：

$$
q=\operatorname{round}_{FP8}(x/s),\qquad \widehat x=sq.
$$

这里 $q$ 是 FP8 解码后的浮点数，不是普通整数编码。在固定指数区间内，还原后的网格间距是 $s\,2^{E-t}$。正规数、最近舍入且无范围问题时，相对舍入误差有常用上界 $2^{-(t+1)}$；这只是单次数值舍入的界，不能当作整个模型的精度损失比例。次正规数、下溢和饱和需要另行分析。

## 4. Scale 和 MXFP8 如何影响精度

Scale 的作用是把数据放到低位宽格式合适的表示范围内。太大的绝对值可能饱和或上溢，太小的绝对值可能进入次正规区或下溢成 0。

如果整个 Tensor 共用一个 scale，少数特别大的值可能迫使其他很小的值落到不利的表示区域。分组量化允许不同组使用不同 scale，减少相距很远的数据互相牵制。

MXFP8 按 **32 个元素一组共享一个 E8M0 scale**。E8M0 的有效有限 scale 是 2 的整数次幂；数据元素本身使用 FP8。定义见 [OCP MX 规范 v1.0，第 5.2、5.4.1 节](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)，本仓库的[量化介绍](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/context/quant_mode_introduction.md)也明确了 E8M0 和 group size 32。

这里需要区分两件事：

- 每组单独选 scale，能更好适应各组的数据范围，缓解部分饱和和下溢。
- FP8 元素的尾数位数没有因此增加；组内差异很大、scale 选择不合适或中间计算再次转换时仍会产生误差。

尤其对于二进制浮点数，在没有上下溢、没有进入次正规区时，单纯乘除 2 的整数次幂不会凭空增加尾数精度。因此不能将“更小的 MX scale”理解成“所有正常数的相对精度必然更高”。

**量化 group 与 FA tile 是两种分组：**32 是共用 scale 的元素数；FA 的 S1/S2 tile 是计算和搬运的任务块。即使都叫 block，也不能把它们当作同一维度或同一大小。具体 Q/K/V 的分组轴可接着看[笔记 03](operator-map.md)。

## 5. 在 FA 源码中，误差出现在哪些环节

先从普通形式 $S=\gamma QK^T$、$P=\operatorname{softmax}(S)$、$O=PV$ 理解：

```text
原始 Q/K/V
   ↓ 量化、还原后的数值可能已经发生改变
QK^T → scale / mask / PSE → 在线 softmax
   ↓ 当前块指数权重转换成较低精度，供 MM2 使用
PV → 用 expMax 修正历史输出 → 最后除以 sum
   ↓ 最终输出类型转换
attention_out
```

源码依据如下，均为实际分支示例，并不表示所有模板都用相同的格式：

| 代码位置 | 可以确认的行为 | 对精度的含义 |
| --- | --- | --- |
| [flash_attention_noquant_block_vec_base.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h) 第 1048–1050 行，`ProcessVec1Nd` | max/sum/exp 局部张量在此路径按 float 使用 | 输入格式不能代表所有中间状态的格式 |
| [vf_basic_block_aligned128_update.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/common/op_kernel/arch35/vf/vf_basic_block_aligned128_update.h) 第 220–227 行 | 用 float 寄存器计算指数与当前块 sum | 归约状态与供 MM2 使用的数据可以有不同精度 |
| 同文件第 264–265、296–297 行 | 按模板将指数权重 Cast 到 BF16 或 FP8 E4M3 | 转换可能改变权重数值；前面保留了 float sum 也无法消除这一步误差 |
| [vf_flashupdate_new.h](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/common/op_kernel/arch35/vf/vf_flashupdate_new.h) 第 33–38、55–64 行 | 用 float 寄存器更新历史输出累积 | 高精度累积有助于控制后续误差，但不能恢复低精度输入丢失的信息 |
| [arch22 的输出转换](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h) 第 2366–2369 行 | 中间类型与输出类型不同时执行 Cast | 最终存储格式也可能引入额外舍入 |

这里实际转换的是块内未最终归一化的 $\widetilde P$，不能据此假设内核先算完整归一化 $P$ 再量化。量化分支还可能先对权重乘 scale，输出阶段再消去相应尺度。

**算子最终输出是 BF16，不代表它等价于从头到尾用 BF16 计算；输入是 FP8，也不代表 max、sum 和输出累积都使用 FP8。**具体乘法与累加行为还需看选中的 Cube 实现，不能只凭类型名推断全部硬件计算细节。

## 6. 为什么输入误差会影响最终输出

以下为实数公式的误差推导，先不考虑额外的硬件舍入。令还原后的输入为 $\widehat Q=Q+\Delta Q$、$\widehat K=K+\Delta K$，则在未屏蔽位置：

$$
\Delta S=\gamma\left(\Delta QK^T+Q\Delta K^T+\Delta Q\Delta K^T\right).
$$

因此 Q、K 的误差会改变 score。对于一行 softmax，当 $\Delta S$ 足够小时，一阶变化为：

$$
\Delta p_j\approx p_j\left(\Delta s_j-\sum_k p_k\Delta s_k\right).
$$

这说明误差如何影响权重取决于分数分布；并不是所有 score 误差都被同等放大。一行所有 score 同加一个常数时，softmax 甚至完全不变，这也解释了为什么减最大值不改变数学结果。

如果最终使用的有效权重为 $\widehat P=P+\Delta P$，Value 为 $\widehat V=V+\Delta V$：

$$
\widehat O-O=\Delta P\,V+P\,\Delta V+\Delta P\,\Delta V.
$$

$\Delta P$ 可以包含 score 变化带来的误差和权重转换误差；再叠加实际归约及输出 Cast 的舍入，才是完整算子误差。因此只测单个 FP8 数值的误差，不能直接断言整个 FA 输出的误差。

## 7. 与在线 Softmax 更新的关系

[笔记 06](../03-attention/online-softmax.md)证明了，在相同数值输入、精确实数运算下：

$$
m_{new}=\max(m_{old},m_{block}),\qquad
\alpha=e^{m_{old}-m_{new}},
$$

$$
l_{new}=\alpha l_{old}+l_{block},\qquad
u_{new}=\alpha u_{old}+u_{block}
$$

可以保持分块前后的数学结果一致。这里的当前块统计量必须已经使用新最大值作为基准。

这项性质回答的是“拆块有没有改变目标公式”。量化精度回答的是“实际存储与计算出来的数值离目标有多远”。正确更新 `expMax` 是正确实现 FA 的必要环节，但不等于量化误差为零。

即使只改变 tile，数学公式不变，浮点求和顺序也可能不同。如果 tile 的改变还影响动态量化的分组或 scale 选择，量化后的输入/中间量也可能改变。

## 8. 如何判断“量化精度够不够”

首先说明参考对象：

1. **衡量量化损失：**比较原始高精度输入的参考输出与量化路径输出；保持相同 scaleValue、mask、PSE、稀疏集合和 dropout 语义。
2. **检查实现正确性：**用相同量化数据、scale、分组和中间量转换约定构造参考计算，检查是否符合选定算子的精度标准。这能帮助区分预期量化误差与索引、scale 广播或更新公式错误。

对输出 $\widehat O$ 和参考 $O$，常用指标有：

$$
\mathrm{MaxAbs}=\max_i|\widehat O_i-O_i|,
\qquad
\mathrm{RMSE}=\sqrt{\frac1N\sum_i(\widehat O_i-O_i)^2},
$$

$$
\mathrm{RelativeL2}=\frac{\|\widehat O-O\|_2}{\max(\|O\|_2,\varepsilon)}.
$$

逐元素允许误差也常表示为：

$$
|\widehat O_i-O_i|\leq atol+rtol\,|O_i|.
$$

参考值接近 0 时，单独使用相对误差容易产生误导；应结合绝对误差和整体指标。`atol` / `rtol` 应来自该算子的测试规范和业务要求，不能只根据 FP8 位数随意给一个百分比。

最后仍需在真实模型任务中检验效果。较小的张量误差与模型任务评分是不同层面的证据，二者之间没有固定换算关系。
