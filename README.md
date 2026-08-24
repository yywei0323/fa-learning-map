# FA / FIA 推理算子开发学习地图

> 面向从零开始学习 **大模型 Attention、FlashAttention、FIA 与昇腾 Ascend C 推理算子开发** 的仓库。

## 目标不是泛学算子，而是看懂并调通推理 FIA

主线分为五步：

1. 大模型基础概念、Transformer 架构与 FA 原理；
2. 推理 FIA 代码架构，以及 Host Tiling / Kernel 关键流程；
3. 使用 aclnn Fuzz 与 Ascend C 调试工具调通一个 FIA V3 用例；
4. 理解 FIA 特性范围，以及 GQA、PA、MLA 等推理特性；
5. 熟悉 Ascend C 基础/高阶 API，理解 Softmax 与 SoftmaxFlash、SoftmaxFlashV2。

## 建议从这里开始

1. [学习总框架：从 Transformer 到 FIA Kernel](docs/01-learning-framework.md)
2. [五阶段学习路线与验收标准](docs/roadmap.md)
3. [FA 零基础扫盲](docs/00-fa-primer.md)
4. [术语表](docs/glossary.md)

## 一张图理解学习对象

```text
大模型生成任务
  └─ Transformer Decoder
      ├─ Attention
      │   ├─ Prefill：一次处理整段 Prompt
      │   └─ Decode：逐 Token 生成 + 读取 KV Cache
      └─ MLP / MoE
           ↓
FlashAttention：Attention 的 IO-aware 分块算法
           ↓
FIA（FusedInferAttentionScore）：昇腾推理融合 Attention 算子
           ↓
aclnn 接口 → Host/Tiling → Kernel/Ascend C → AI Core
```

## 仓库规划

```text
fa-learning-map/
├── README.md
├── docs/
│   ├── 00-fa-primer.md
│   ├── 01-learning-framework.md
│   ├── roadmap.md
│   └── glossary.md
├── notes/          # 后续：源码阅读和调试笔记
├── examples/       # 后续：Python参考实现、FIA用例
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

- [CANN ops-transformer：Transformer 类大模型算子库](https://gitcode.com/cann/ops-transformer)
- [FusedInferAttentionScore 源码目录](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)
- [Ascend C 自定义算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)
- [SoftmaxFlash](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0756.html)
- [SoftmaxFlashV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0758.html)
- [Ascend C 调测工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasascendebug_16_0077.html)
