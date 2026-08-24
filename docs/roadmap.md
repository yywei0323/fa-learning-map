# 面向推理 FIA 的五阶段学习路线

## 总目标

不是从头实现一个工业级 FIA，而是先具备以下闭环能力：

```text
理解模型语义 → 找到接口与代码 → 看懂 Tiling → 跟踪 Kernel
→ 构造用例 → 编译运行 → 定位问题 → 判断特性分支
```

## 阶段 1：大模型、Transformer 与 FA 原理

### 学习内容

1. 大模型基础：
   - Token、Embedding、参数量、训练与推理；
   - 自回归生成；
   - Prefill 与 Decode；
   - KV Cache 为什么存在。
2. Transformer：
   - Decoder Block；
   - RMSNorm/LayerNorm、Residual；
   - Q/K/V 投影；
   - Attention；
   - MLP / MoE；
   - RoPE。
3. Attention 与 FA：
   - Scaled Dot-Product Attention；
   - MHA；
   - 稳定 Softmax；
   - 标准 Attention 的中间矩阵与 IO 瓶颈；
   - Tiling、Online Softmax、融合；
   - FA1 与 FA2 的思想差异。

### 实践

- 用一个极小 Shape 手算 Attention；
- 写 PyTorch 参考实现；
- 将 KV Cache 加入逐 Token Decode；
- 写一个分块 Online Softmax 参考实现。

### 验收

能够回答：

- Prefill 和 Decode 的输入 Shape、计算特点有什么不同？
- KV Cache 节省了什么，又增加了什么成本？
- FA 为什么快，而不是只说“因为做了融合”？
- Online Softmax 的 max、sum 和历史输出如何更新？

## 阶段 2：推理 FIA 代码架构与关键流程

### 源码阅读顺序

不要从最大的 Kernel 模板开始。按调用链阅读：

1. 算子 README 与 aclnn 接口文档；
2. aclnn 参数与输入输出；
3. 算子原型、Shape/类型推导；
4. Host Tiling 入口；
5. 参数校验与特性识别；
6. Tiling Key / 模板选择；
7. TilingData 定义；
8. Kernel 统一入口；
9. 选定的一个 Kernel 模板；
10. BMM1 → Mask/Softmax → BMM2 → 输出。

### Tiling 要看懂什么

- 输入 Shape 和 Layout 如何解释；
- Q 头数、KV 头数、序列长度、Head Dim；
- 如何分配 AI Core；
- Q 与 KV 分块大小；
- UB/L1/Workspace 预算；
- 尾块与对齐；
- 不同特性如何选择 Tiling Key；
- Host 生成的字段在 Kernel 哪里消费。

### Kernel 主流程

```text
读取 TilingData
  → 定位当前 Core 的 Batch / Head / Query 块
  → 搬入 Q
  → 遍历 K/V 块
      → BMM1: Q × Kᵀ
      → Scale / PSE / Mask
      → Online Softmax
      → BMM2: P × V
      → 更新历史 max、sum、输出累加
  → 最终归一化
  → 写回 Attention Output / 可选 LSE
```

### 验收

选定一个最简单 FIA 用例，能从 aclnn 输入一路指到实际 Kernel 模板，并画出关键参数的数据流。

## 阶段 3：调通一个 FIA V3 用例

### 范围控制

第一个用例应尽量简单：

- 单 Batch；
- BNSD 或团队基线 Layout；
- FP16/BF16；
- 普通 MHA；
- 无量化；
- 无 PA；
- 无 MLA；
- 先无复杂 Mask；
- 使用仓库已有且已知支持的 Shape。

### 调试闭环

1. 用 aclnn Fuzz 框架生成或选择用例；
2. 保存输入参数、随机种子、Shape、Layout、dtype；
3. 确认进入 FIA V3 aclnn 接口；
4. 运行 Host 参数校验与 Tiling；
5. 记录 Tiling Key、blockDim、workspace、TilingData；
6. 编译 Kernel；
7. 使用 Ascend C Debug Tool / msDebug / 调试宏定位；
8. 检查 Kernel 输出；
9. 与 Golden 结果比较；
10. 固化为可重复回归的 Case。

### 推荐排错顺序

```text
接口参数错误
→ Shape / Layout / dtype 不支持
→ Host 校验失败
→ Tiling 失败
→ Kernel 编译失败
→ 运行越界 / 同步问题
→ 结果不一致
→ 性能问题
```

### 验收证据

必须留下：

- 完整环境版本；
- 可复现命令；
- 用例配置；
- Tiling Key 与核心 TilingData；
- Golden 比较结果；
- 一份调试记录。

> 这一步的具体命令不能脱离你实际使用的代码分支和内部/开源 Fuzz 工程编造。拿到环境信息后再建立可执行 Runbook。

## 阶段 4：FIA 特性范围

### 先建立“算法、存储、工程”三层分类

| 特性 | 主要解决什么 | 对 FIA 的典型影响 |
|---|---|---|
| MHA | 标准多头注意力 | Q/K/V 头数一致，基础路径 |
| GQA | 多个 Q 头共享较少 KV 头 | 头映射、KV 复用、核间切分变化 |
| MQA | 所有 Q 头共享一组 KV | KV 带宽更低，头映射更特殊 |
| PA | KV Cache 分页管理 | block table、page/block size、非连续地址访问 |
| MLA | KV 低秩压缩与解耦表示 | Q/K/V 维度与布局不再是普通 MHA 直觉 |
| Quant | 减小带宽与存储 | scale/offset、反量化、累加精度与路径选择 |
| Sparse | 只计算部分注意力区域 | mask/索引、块选择、实际计算量改变 |
| Variable Length | Batch 内序列长度不同 | actualSeqLen、空块、负载均衡 |

### 必须理解的关系

- GQA 是注意力头的组织方式；
- PA 是 KV Cache 的存储与寻址方式；
- MLA 是注意力表示与缓存压缩架构；
- 它们不是互斥概念，实际用例可能组合出现。

### 验收

给定一组 FIA 参数，能判断它属于普通 MHA、GQA、PA 或 MLA 中的哪些组合，并指出代码中大概率影响 Host 还是 Kernel。

## 阶段 5：Ascend C API 与 Softmax

### 基础 API

重点不是背完整手册，而是能认出：

- 数据搬运：DataCopy 及相关增强搬运；
- 逐元素计算：Add、Sub、Mul、Div、Exp；
- 归约：ReduceMax、ReduceSum；
- 广播、格式转换与类型转换；
- 同步、Event、Queue/Pipe；
- LocalTensor、GlobalTensor 与内存管理；
- 调试输出与断言。

### 高阶 API

重点：

- Matmul；
- SoftMax / SimpleSoftMax；
- SoftmaxFlash；
- SoftmaxFlashV2；
- 对应的 Tiling API；
- 高阶 API 所需临时空间、Shape 和格式约束。

### Softmax 原理

稳定 Softmax：

```text
m = max(x)
e = exp(x - m)
l = sum(e)
p = e / l
```

分块更新的关键：

```text
m_new = max(m_old, max(x_block))
alpha = exp(m_old - m_new)
l_new = alpha * l_old + sum(exp(x_block - m_new))
```

历史输出累加也必须乘 alpha，与新块统一到相同的指数尺度。

### SoftmaxFlash 与 SoftmaxFlashV2

| 项目 | SoftmaxFlash（团队可能称 V1） | SoftmaxFlashV2 |
|---|---|---|
| 官方定位 | SoftMax 增强接口 | SoftmaxFlash 增强版，对应 FA2 |
| 分块更新 | 支持 | 支持 |
| 普通分支输出 | 返回归一化结果 | 主要输出未归一化指数结果及 max/sum，便于后续融合 |
| 数据格式 | 官方旧文档说明仅 ND | 支持范围更广，具体看产品与版本 |
| 状态 | 官方文档提示后续废弃 | 官方建议使用，精度和性能更好 |
| 工程使用 | 先读旧代码时理解 | 新开发重点 |

真正读代码时，需要逐项对照所用 CANN 版本的函数重载、模板参数、临时空间与输出语义，不能只凭 V1/V2 名字猜测。

### 验收

- 能手写稳定 Softmax 与 Online Softmax；
- 能解释 FIA 中为什么不一定立即除以 sum；
- 能顺着一次 SoftmaxFlashV2 调用找到 Host Tiling 和 Kernel 输入；
- 能判断何时应使用高阶 API，何时需要基础 API 组合。

## 推荐学习顺序

```text
第 1 周：大模型推理 + Transformer + Attention
第 2 周：FA + Online Softmax + Python Reference
第 3 周：FIA 接口、目录、Host/Tiling、Kernel
第 4 周：调通一个 FIA V3 基线用例
第 5 周：GQA / PA / MLA 与特性分支
第 6 周：Ascend C API + SoftmaxFlashV2 深入
```

时间只是建议，真正进入下一阶段的依据是“验收标准是否达成”。
