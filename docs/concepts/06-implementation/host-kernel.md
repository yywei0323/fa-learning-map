# 第 6 层：从接口到 Host、Tiling、Kernel

[概念目录](../README.md) · [源码学习框架](../../01-learning-framework.md)

**算子定义数学与接口契约，Host 准备执行，Kernel 执行主体计算。** 一个算子可以有多个 Kernel 或辅助算子；目录名也不保证一条统一调用链。

## 职责层级

| 层级 / 名词 | 主要职责 | 阅读时问什么 |
|---|---|---|
| Framework / PyTorch 接入层 | 构造输出、传递 Tensor/属性、调用底层入口 | eager 和 Meta 是否遵守相同 shape 约定？ |
| Operator / 算子原型 | 声明输入、输出和属性约束 | shape、dtype、layout、可选参数是什么？ |
| aclnn | CANN 单算子调用接口的一类入口 | 具体版本、workspace、executor 和 stream 怎样使用？ |
| Host / op_host | 解析、校验、推导和准备执行 | 本次 case 是否支持，走哪条实现？ |
| InferShape / InferDataType | 推导输出形状 / 数据类型 | 输出描述与接入层分配是否一致？ |
| Tiling | 准备分块、分核、实现选择与资源参数 | 块大小、核数、尾块、缓冲容量怎样确定？ |
| TilingData | 本次执行的数据参数 | 哪个字段在哪里填写、在哪里读取？ |
| TilingKey | 选择实现或模板组合的标识 | 声明、编码、Kernel 实例是否匹配？ |
| Device / op_kernel | 搬运、计算、同步、写回 | 本核算哪一块，每个地址怎样得到？ |
| Metadata | 某些实现额外准备的调度描述 | 谁生成、描述哪些任务、何时被消费？ |
| tests | 输入构造、参考输出、执行和检查 | 测到哪个分支，覆盖什么边界？ |

## 几个常见运行参数

**Workspace** 是一次执行需要的临时空间，如部分结果和跨阶段数据；它与跨 Decode 步骤持久保存的 KV Cache 不同。**Stream** 是设备任务的有序提交与执行上下文，异步提交不代表结果此时已经完成。**blockDim / coreNum** 是启动配置或实现字段；在不同任务类型中其解释要结合运行模式，不能总等同于物理总核数。

Tensor 的描述包括 shape、stride、dtype、device 等，数据内容是实际数值。InferShape 通常从描述和属性推导，有些算子还存在数据依赖；不能由“形状推导”推断从不访问输入值。

## 一次阅读的最小闭环

```text
输入契约和具体 case
  → 接入层输出构造
  → 解析、校验、形状/类型规则
  → Tiling 选择 Key、填写 Data 和启动资源
  → 对应 Kernel 模板读取参数
  → 找到本核任务与 Q/K/V 地址
  → MM1 → Mask/Softmax → MM2 → 更新/写回
  → 同语义 Golden 比较
```

这是一条阅读顺序，不是声称所有框架模式都按完全相同的函数调用顺序执行。尤其 Query 任务调度可能依赖另一个 metadata 算子，不能只看 Tiling 中的一个函数。

独立词条：[InferShape](InferShape.md) · [Tiling](Tiling.md) · [TilingData](TilingData.md) · [TilingKey](TilingKey.md)。

源码依据：[AI Core 开发指南](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/docs/zh/develop/aicore_develop_guide.md)、[QBSA Host Tiling](https://gitcode.com/cann/ops-transformer/blob/1fe74acf0a60fea0ed143c3cda8054f6c373063f/experimental/attention/quant_block_sparse_attn/op_host/quant_block_sparse_attn_tiling.cpp)。
