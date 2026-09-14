# 来源、版本与覆盖范围

[概念目录](README.md)

整理日期：2026-09-14。目标仓库：**yywei0323/fa-learning-map**。此目录是学习文档整理，未修改或向 ops-transformer 源码仓库提交内容。

## 已纳入的个人学习资料

| 来源 | 覆盖内容 | 本次处理 |
|---|---|---|
| ChatGPT 项目“fa算子开发学习”的可读取对话 | 模型、推理、Attention、硬件、Tiling、会议术语、环境 | 按概念归类，没有发布原始对话全文 |
| 学习仓库已有文档 | FA 扫盲、学习框架、LLM 基础、PyTorch 导读、端到端 Transformer、路线、术语表 | 保留课程，新增概念目录和交叉链接 |
| 本地 note/名词解释 | BNSD、BSND、TND、layout、InferShape、Tiling、TilingData、TilingKey、LSE 与关系记录 | 保留详细词条，转换成学习仓库相对链接与固定源码引用 |
| 本地 note/01～07 | 工程规范、Host/Device、MXFP8 布局、设计/PR 对照、在线 Softmax、量化精度 | 基础概念重组；Softmax 和精度推导保留详解；设计/审查收敛成版本案例 |
| 本地保留的算子源码与 PR 快照 | API、分组、地址、模板选择和更新公式 | 用作版本特定事实的核对依据，没有复制整个源码仓 |

已读取的 5 个项目对话：**FA算子学习入门、解析Transformer源码、开发环境选择比较、科普Cube与Vector计算、解释 Tiling 流程**。这些是话题覆盖证据，不用于断言学习者已经掌握相关知识。

## 尚未取得的材料

本次打开 ChatGPT 网页项目资料时连续超时，现有项目/对话读取接口没有返回可下载的项目级独立文件。因此**没有把“已读取项目对话”表述成“已下载全部项目附件”**。未读取附件的文件名、内容和完整性尚待补齐。

收到可读取附件后，应对照本页和术语索引去重、补遗漏、保留原记录的版本；不要直接追加一份平行且互相矛盾的术语表。

## 固定版本

| 对象 | 版本 |
|---|---|
| 本次整理起点：学习仓库 main | 61b863b2dccd4525164c92b3c52409e88e3d7948 |
| 本地算子学习基线 | 1fe74acf0a60fea0ed143c3cda8054f6c373063f |
| QBSA layout PR #11704 head | ee2a4c58086472d1297ae2f4a7174c9561b7bfe8 |
| PR diff base | 125200e0afd4b13719694a9014e6053b89e1ad09 |
| PR 快照审查日期 | 2026-09-10 |

详解中“当前、本地、尚未支持”均限定于所注明的基线。独立的 BNSD/BSND 扩展案例使用 PR head，不能与基线混读。源码链接从个人电脑路径改成固定 commit 的上游路径；其正文依据已保留的本地文件，并非声称所有网页链接本次均成功在线打开。

## 外部概念核对资料

| 材料 | 使用范围 / 访问情况 |
|---|---|
| [FlashAttention 论文](https://arxiv.org/abs/2205.14135) | IO-aware、分块和数学目标；本次读取摘要核对 |
| [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691) | 并行划分和非矩阵乘开销；本次读取摘要核对 |
| [Hugging Face KV Cache](https://huggingface.co/docs/transformers/cache_explanation) | 缓存动机和逐步增长；本次读取 |
| [FP8 Formats for Deep Learning](https://arxiv.org/html/2209.05433v2) | E4M3/E5M2 字段和数值表示；本次读取 |
| [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | 原量化笔记引用的规范；本次远程取回失败，MX 分组同时以本地量化介绍和算子接口核对 |
| [PyTorch Tensor 视图](https://docs.pytorch.org/docs/stable/tensor_view.html) | view、reshape、transpose、contiguous 的官方核对入口 |
| [项目指定 CANN 编程指导](../references/cann-ascendc-guide.md) | 保留原入口；旧链接本次未返回对应硬件/API 正文，不据此填写机器容量或接口支持表 |

既有课程的 Happy-LLM、LLMs-from-scratch 等参考继续留在对应章节，本次没有整篇复制外部教材。

## 验证边界

本次文档校验针对新增/修改的本仓相对链接、术语索引和示例公式。在线 Softmax 的教学脚本保留在 [examples/concepts](../../examples/concepts/verify_online_softmax.py)，验证实数近似的 CPU 数学模型，不验证 Ascend Kernel、MXFP8 硬件精度或性能。已有 Notebook 的历史运行状态仍以原课程记录为准。

本次重新执行教学脚本：526 种分块全部通过，最大输出绝对差约 6.217e-15；两块示例输出为 34.926527346。该结果只对应 Python 双精度数学模型。
