# FA 算子开发术语表

| 术语 | 零基础解释 |
|---|---|
| Operator / 算子 | 计算图中的一个计算单元，同时包含数学语义、接口和硬件实现。 |
| Kernel / 核函数 | 真正在加速器上执行的程序。一个算子可能包含一个或多个 Kernel。 |
| Host 侧 | 在 CPU 侧完成算子定义、Shape 推导、Tiling 计算、启动准备等工作。 |
| Device / Kernel 侧 | 在 NPU AI Core 上执行数据搬运与计算的部分。 |
| CANN | 昇腾异构计算软件栈，包含编译、运行时、算子库和开发工具等。 |
| Ascend C | 面向昇腾 AI 处理器的算子编程语言与编程体系，语法基于 C/C++。 |
| AI Core | 昇腾芯片上执行 AI 计算的核心。 |
| GM | Global Memory，可理解为容量大但离计算单元较远的全局内存。 |
| Local / 片上存储 | 更靠近计算单元、速度快但容量有限的存储空间。具体层级依芯片和编程接口而异。 |
| DataCopy | 在不同存储层次之间搬运数据。 |
| Tiling | 将大问题切成适合片上容量和硬件计算粒度的小块。 |
| Tile | Tiling 后的一个数据块。 |
| Shape | 张量各维度的大小。 |
| Layout | 张量各语义维度的排列与物理组织方式。 |
| M/N/K | 矩阵乘 `[M,K] × [K,N]` 中的三个核心维度。 |
| Alignment / 对齐 | 数据起始地址、长度或计算粒度满足硬件要求。 |
| Tail / 尾块 | 总长度不能被块大小整除时，最后剩余的不完整块。 |
| Pipeline / 流水 | 将搬运、矩阵计算、向量计算等阶段重叠起来。 |
| Double Buffer | 准备两份缓冲区，计算当前块的同时预取下一块。 |
| Fusion / 融合 | 将多个计算阶段合到更少的 Kernel 中，减少中间数据落地和启动开销。 |
| Attention | 根据 Q 与 K 的相关性，对 V 做加权汇总。 |
| MHA | Multi-Head Attention，多头注意力。 |
| GQA | Grouped-Query Attention，多组 Query 头共享较少的 KV 头。 |
| MQA | Multi-Query Attention，多个 Query 头共享同一组 KV 头。 |
| FA | FlashAttention，一类关注内存 IO 的精确 Attention 算法与实现。 |
| Online Softmax | 分块读取数据时，增量维护全局最大值与指数和，从而得到全局 Softmax。 |
| Mask | 屏蔽不应参与注意力的元素，如 causal mask、padding mask。 |
| Causal Mask | 让当前位置不能看到未来位置的 mask。 |
| FP16 | 16 位浮点格式，范围与精度有限。 |
| BF16 | 16 位浮点格式，指数范围接近 FP32，但尾数精度较低。 |
| FP32 Accumulation | 输入可以是 16 位，但归约或累加用 32 位提高稳定性。 |
| FLOPs | 浮点运算次数，只描述计算量，不等于实际运行时间。 |
| Bandwidth / 带宽 | 单位时间可搬运的数据量。 |
| Latency / 时延 | 一次算子执行耗时。 |
| Throughput / 吞吐 | 单位时间处理的数据或计算量。 |
| Profiling | 采集执行时间、硬件利用率、流水等待等性能数据。 |
| Golden / Reference | 用高可信实现产生的参考结果，供精度对比。 |
| atol / rtol | 绝对误差与相对误差阈值。 |
