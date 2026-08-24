# FA 算子开发学习路线

目标：从“看懂名词”走到“能解释、实现、验证并优化一个简化 FA 算子”。

## 阶段 0：补齐最低前置知识

### 需要会

- C/C++：函数、类、模板基础、指针、内存、编译链接；
- Python：NumPy/PyTorch 张量操作；
- 线性代数：矩阵乘、转置、广播、归约；
- 深度学习：Attention、Softmax、MHA；
- Linux：Shell、环境变量、CMake 基础。

### 达标标准

能手算一个很小的 Attention，并用 Python 写出参考实现。

## 阶段 1：认识 Ascend C 算子工程

学习：

- CANN、Ascend C、AI Core；
- Host/Kernel 分工；
- GM 与片上存储；
- 核函数启动、核间并行；
- DataCopy、队列、基础矢量计算；
- CPU/NPU 调试与精度验证。

练习顺序：

1. Add；
2. ReduceSum / ReduceMax；
3. Exp；
4. 简化 Softmax。

达标标准：能独立解释一个 Add 样例中数据从哪里来、在哪里算、写到哪里。

## 阶段 2：掌握矩阵计算与 Tiling

学习：

- MatMul 的 M/N/K；
- 分块尺寸与对齐；
- 尾块处理；
- 多核任务切分；
- Layout 与转置；
- Tiling 参数从 Host 传到 Kernel。

练习：固定 Shape MatMul → 动态 Shape MatMul。

达标标准：改变 Shape 后，能够解释 Tiling 为什么变化。

## 阶段 3：掌握稳定 Softmax 与 Online Softmax

学习：

- max-subtraction；
- 行归约；
- FP32 累加；
- 分块状态 `m`、`l`；
- 历史结果重标定。

练习：

1. 普通稳定 Softmax；
2. 两块 Online Softmax；
3. 任意块数；
4. 加 causal mask；
5. 与 PyTorch 高精度参考对比。

达标标准：不查资料也能写出 Online Softmax 状态更新逻辑，并说明为什么等价。

## 阶段 4：实现最小 FA Forward

限定条件：

- 单 Batch、单 Head；
- 固定长度；
- 固定 Head Dim；
- 无 dropout；
- 先无 mask，再加 causal mask；
- FP16/BF16 输入，较高精度累加。

数据流：

1. 加载 Q 块；
2. 循环加载 K/V 块；
3. 计算分数块；
4. scale + mask；
5. Online Softmax 更新；
6. 当前概率块与 V 相乘；
7. 更新输出累加；
8. 归一化并写回。

达标标准：多组小 Shape 与参考实现一致，无越界、NaN 和 Inf。

## 阶段 5：工程化

逐步增加：

- Batch、多 Head；
- MHA/GQA/MQA；
- 多种 Layout；
- 变长序列；
- 多种 mask；
- 动态 Shape；
- 多种数据类型；
- PyTorch/torch_npu 调用；
- 算子原型、Shape 推导、部署交付。

达标标准：不仅 Kernel 能运行，而且接口、编译、部署、调用和测试闭环完整。

## 阶段 6：性能优化

按照证据推进：

1. 建立可靠基线；
2. Profile；
3. 判断瓶颈在矩阵、向量、搬运还是同步；
4. 一次只改一类因素；
5. 回归正确性与精度；
6. 记录 Shape、配置、耗时和结论。

常见方向：

- 更合理的 Q/K/V 分块；
- 减少 GM 读写；
- 提高片上数据复用；
- Double Buffer；
- 搬运与计算流水并行；
- 矩阵与向量阶段并行；
- 减少同步、转置和格式转换；
- 改善多核负载均衡；
- 针对常见 Shape 选择 Tiling 策略。

## 推荐的学习产出

每一阶段都在仓库留下：

- 一页概念笔记；
- 一个能运行的最小例子；
- 一组正确性测试；
- 一张性能记录表；
- 一段“失败原因与修复”的复盘。

不要只收藏链接。可解释、可运行、可验证的内容才算真正掌握。
