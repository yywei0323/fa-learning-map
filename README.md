# FA / FIA 推理算子开发学习地图

> 面向从零开始学习 **大模型 Attention、FlashAttention、FIA 与昇腾 Ascend C 推理算子开发** 的仓库。

## 目标不是泛学算子，而是看懂并调通推理 FIA

主线分为五步：

1. 大模型基础概念、Transformer 架构与 FA 原理；
2. 推理 FIA 代码架构，以及 Host Tiling / Kernel 关键流程；
3. 使用 aclnn Fuzz 与 Ascend C 调试工具调通一个 FIA V3 用例；
4. 理解 FIA 特性范围，以及 GQA、PA、MLA 等推理特性；
5. 熟悉 Ascend C 基础/高阶 API，理解 Softmax 与 SoftmaxFlash、SoftmaxFlashV2。

## 名词解释与概念关联

按知识依赖学习，请先读 [知识网络与分层学习地图](docs/concepts/knowledge-network.md)：包含八层内部学习顺序、跨层关系、六次聚焦学习和练习验收。

新增 [八层概念目录](docs/concepts/README.md)、[全部术语索引](docs/concepts/index.md)和[概念关系图](docs/concepts/relations.md)，集中整理当前可访问学习资料。独立项目附件尚未成功下载，覆盖范围与固定源码版本见[来源记录](docs/concepts/sources.md)。

## 建议从这里开始

1. [第 1 章｜大模型基础：从文本到逐 Token 推理](docs/01-llm-basics/README.md)
2. [第 2 章｜从 Token 到 Loss：端到端读懂 Transformer 源码](docs/02-transformer-end-to-end/README.md)
3. [第 2 章配套 Notebook｜完整前向、Loss、反向传播与 Shape 断言](examples/ch02/transformer_end_to_end.ipynb)
4. [可选补充｜掩码自注意力与多头注意力](docs/01-llm-basics/02-pytorch-transformer-code-guide.md)
5. [学习总框架：从 Transformer 到 FIA Kernel](docs/01-learning-framework.md)
6. [五阶段学习路线与验收标准](docs/roadmap.md)
7. [FA 零基础扫盲](docs/00-fa-primer.md)
8. [术语表](docs/glossary.md)
9. [CANN / Ascend C 编程指导与硬件资料入口](docs/references/cann-ascendc-guide.md)

## 一张图理解学习对象

```text
大模型生成任务
├─ 执行阶段：Prefill / Decode，使用 KV Cache
└─ Transformer Decoder
    ├─ Attention：QKᵀ → Scale/Mask → Softmax → PV
    │   ├─ 算法方法：FlashAttention 的分块、在线更新与 IO 优化
    │   └─ 推理算子实现之一：FIA（FusedInferAttentionScore）
    │       └─ 接口 → Host/Tiling → Ascend C Kernel → AI Core
    └─ MLP / MoE
```

## 仓库规划

```text
fa-learning-map/
├── README.md
├── docs/
│   ├── 00-fa-primer.md
│   ├── 01-learning-framework.md
│   ├── 01-llm-basics/
│   │   ├── README.md
│   │   └── 02-pytorch-transformer-code-guide.md
│   ├── 02-transformer-end-to-end/
│   │   └── README.md
│   ├── concepts/   # 八层名词解释、关系图与版本案例
│   ├── roadmap.md
│   └── glossary.md
├── assets/ch01/    # 第1章教学插图
├── notes/          # 后续：源码阅读和调试笔记
├── examples/
│   ├── ch01/
│   │   ├── pytorch_transformer_shapes.ipynb
│   │   └── pytorch_transformer_shapes.py
│   └── ch02/
│       └── transformer_end_to_end.ipynb
└── benchmarks/     # 后续：精度与性能记录
```

## 重要辨析

- **FA**：FlashAttention 算法家族，不等于某一个具体接口。
- **FIA**：FusedInferAttentionScore，面向推理场景的昇腾融合 Attention 算子。
- **PFA / IFA**：通常分别指 PromptFlashAttention 与 IncreFlashAttention，对应 Prefill 与增量 Decode 侧重点。
- **FIA V3**：应结合当前代码分支理解为 aclnn 接口/实现版本，不能直接等同于 FlashAttention-3。
- 官方 API 中通常叫 **SoftmaxFlash** 和 **SoftmaxFlashV2**；团队代码把前者称为 V1 时，要以实际接口签名为准。

## 版本原则

开始真实调试前必须记录：

- NPU 型号；
- CANN 版本；
- ops-transformer / 算子代码分支或 Commit；
- torch 与 torch_npu 版本；
- 编译器、OS 与 CPU 架构；
- 测试框架版本。

master 代码、CANN 安装包和 API 文档不配套时，很容易出现编译错误、Tiling 失败或运行时错误。

## 主要官方资料

- **项目指定参考**：[CANN 8.3 RC1 Ascend C 编程指导：基本架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)。用户指定其为编程与硬件资料入口，详见[项目记录](docs/references/cann-ascendc-guide.md)。

- [CANN ops-transformer：Transformer 类大模型算子库](https://gitcode.com/cann/ops-transformer)
- [FusedInferAttentionScore 源码目录](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)
- [Ascend C 自定义算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)
- [SoftmaxFlash](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0756.html)
- [SoftmaxFlashV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0758.html)
- [Ascend C 调测工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasascendebug_16_0077.html)
