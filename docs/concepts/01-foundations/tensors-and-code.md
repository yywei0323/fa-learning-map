# 第 1 层：张量与代码基础

[概念目录](../README.md) · [关联关系](../relations.md)

这一层解决“数据是什么、每个轴代表什么、代码怎样表达计算”。复习时用已有的[多头注意力 Notebook](../../../examples/ch01/pytorch_transformer_shapes.ipynb)对照，不需要先背完整 PyTorch API。

## 张量描述

| 术语 | 含义与例子 | 容易混淆的地方 |
|---|---|---|
| Tensor / 张量 | 多维数据及其描述，例如一批 token 的向量 | 不是只指矩阵，也可以是标量或高维数组 |
| Scalar / 标量 | 单个数；张量 shape 为 `[]` | `[0]` 是零元素一维张量，`[1]` 是一元素一维张量 |
| Rank / 维数 | 轴的数量；`[2,4,8]` 的 rank 为 3 | 这里不是线性代数的矩阵秩 |
| Shape | 各轴长度；`[B,S,H]` | 数字相同不能证明轴语义相同 |
| dtype | 元素的数值类型，如 BF16、FP32、INT64 | 不能由 shape 推断 |
| device | 数据所在的设备，如 CPU 或 NPU | Host/Device 也是程序职责划分，见后文 |
| numel | 元素总数；shape 各轴长度之积 | 数据字节数还要乘每元素字节数 |
| stride | 沿某轴移动一步跨过多少元素 | 必须检查接口的步长单位，有的字段以字节为单位 |
| Contiguous / 连续 | 满足对应存储顺序的步长关系 | 交换轴后的 view 往往不连续 |
| Layout | 轴的语义和排列，例如 BNSD | [详细辨析](../04-layout/layout.md)：shape、stride、格式并不等同 |

## 读懂张量操作

| 操作 | 作用 | 结合 Attention 理解 |
|---|---|---|
| Index / Slice | 选择某个位置或一段数据 | `x[:, -1, :]` 取各 batch 最后一个位置的向量 |
| Broadcasting / 广播 | 从尾轴对齐，长度相等或其中一个为 1 的轴可扩展参与运算 | `[B,N,Sq,Skv]` 的 score 加上 `[B,1,1,Skv]` 的 padding bias |
| view / reshape | 重新解释 shape，元素逻辑顺序保持不变 | 拆 `H=N*D`；view 有步长兼容要求，reshape 必要时会复制 |
| transpose / permute | 交换或重排轴 | `[B,S,N,D]→[B,N,S,D]`；常先改变视图而不搬数据 |
| contiguous | 必要时复制成连续布局 | 交换 N/S 后再合头时常见 |
| unsqueeze / squeeze | 添加或去掉长度为 1 的轴 | 让 mask 具备可广播的维度 |
| matmul / @ | 最后两个轴做矩阵乘，前面是批量维 | Q 的 D 必须与转置 K 的归约维匹配 |
| Elementwise | 逐元素计算 | scale、加法、Exp、Cast 等 |
| Reduce / 归约 | 沿指定轴汇总 | Softmax 在 KV 轴做 max、sum |
| Cast | 改变数据类型 | FP32 指数权重转 BF16/FP8，可能损失数值信息 |

`reshape` 不等于“任意换轴”。BSND 变 BNSD 应交换 S/N；只改 shape 即使元素数相同，也可能把 token 与 head 的对应关系弄错。

## PyTorch 模块与训练代码

| 名词 | 最小解释 |
|---|---|
| nn.Module、forward、__init__ | Module 组织模型；__init__ 创建子模块和状态；forward 定义本次数据流 |
| nn.Parameter | 注册可学习张量，使优化器能够找到它 |
| register_buffer | 注册随模块保存、迁移但不作为普通可学习参数的状态，例如位置编码 |
| ModuleList / ModuleDict | 注册多层子模块；普通 Python 容器不能替代所有模块注册行为 |
| nn.Embedding | 根据 token ID 查向量表 |
| nn.Linear | 在最后一维做线性映射；PyTorch 权重存储为 [out_features,in_features] |
| Autograd、backward、grad | 自动求导、触发反向传播、保存参数梯度 |
| train / eval | 切换 Dropout 等模块的训练/推理行为 |
| no_grad / inference_mode | 控制梯度记录等执行开销；与 eval 的职责不同 |
| state_dict | 模块参数和持久 buffer 的命名状态集合 |

详细数据流见[端到端 Transformer 课程](../../02-transformer-end-to-end/README.md)。本页是已学代码的索引，不把它等同于完成全部框架机制学习。

## C++ 与工程词汇

| 名词 | 在算子开发中的意思 |
|---|---|
| Compile time / Runtime | 编译时决定程序结构；运行时处理本次地址、shape、尾块等 |
| Template / 模板 | 用类型或常量参数生成具体实现；例如按 dtype、layout 实例化 |
| if constexpr | 编译时选择保留的分支 |
| struct / enum | 结构体聚合字段；枚举为有限配置命名 |
| Pointer / Offset | 指向数据的位置，以及相对起点的偏移 |
| Namespace、Header、Macro | 组织名称、声明/模板代码、预处理展开；宏名本身不是算法证明 |
| CMake / 编译器 / 链接器 | 配置构建、编译源文件、组合产物 |
| Git Commit / Branch / PR | 固定代码快照、演进分支、供审查合入的一组变更 |
| Notebook | 文本、公式、代码与输出放在一起的实验文档 |

[模板怎样用于 Matmul](../06-implementation/scheduling-and-matmul.md) · [环境和测试术语](../08-validation/performance-and-tests.md)

补充核对入口：[PyTorch Tensor 视图](https://docs.pytorch.org/docs/stable/tensor_view.html)、[nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。已有学习来源和版本见[来源清单](../sources.md)。
