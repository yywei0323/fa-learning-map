# CANN / Ascend C 编程指导与硬件资料入口

## 项目参考资料

- 用户指定资料：[Ascend C 编程指导：基本架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- 文档版本：链接路径为 `83RC1`，页面标题标注为 CANN 8.3.RCX 开发文档。
- 章节位置：算子开发 → Ascend C 算子开发 → 概念原理和术语 → 硬件架构与数据处理原理 → 基本架构。
- 项目约定：将本指导作为后续 Ascend C 编程、硬件架构及 FA/FIA 算子学习的重要官方参考。用户明确说明，项目相关硬件信息也在这份指导中。
- 代码参考：[ops-transformer](https://gitcode.com/cann/ops-transformer)；[用户指定的 flash_attn 目录](https://gitcode.com/cann/ops-transformer/tree/master/attention/flash_attn)。

## 后续学习如何使用

| 学习问题 | 对照阅读与记录重点 |
| --- | --- |
| Cube / Vector 的职责 | 对应计算单元、支持的运算与协作方式 |
| 数据搬运与片上存储 | 对应硬件的数据通路、存储层次与容量 |
| Host Tiling | 结合实际平台资源理解分块、分核、缓冲区和 workspace |
| Kernel 执行 | 将矩阵乘、Softmax、数据搬运和同步对应到硬件执行过程 |
| layout / stride | 结合输入布局理解地址计算与数据搬运 |
| QBSA-MXFP8 / Stem Indexer / FIA | 分别核对目标硬件、数据类型、API 与源码版本的支持范围 |

以上为项目阅读索引，不表示已核验该页面包含所有具体 API 或容量参数。

## 硬件与版本记录原则

用户指定本指导为项目硬件资料入口。后续解释必须结合文档内对应硬件型号的章节，避免把不同产品的核数、存储容量或能力混用。

| 项目 | 当前记录 |
| --- | --- |
| 编程指导版本 | 用户提供的 `83RC1` 链接 |
| 已有 QBSA 编译目标 | 此前提供的编译命令包含 `--soc=ascend950` |
| 实际设备具体型号、核数和存储容量 | 尚未从本次页面读取中核实；编译目标不等同于设备实测信息 |
| 实际安装的 CANN 版本 | 尚未核实；不能由参考文档版本直接推定 |
| 本地 ops-transformer 分支 / commit | 后续源码讲解或调试时记录 |

## 本次读取范围

已核对链接及页面标题，确认它是硬件架构下的“基本架构”入口。本次网页读取仅获得站点导航与页面标题，未取得硬件正文和架构图，因此本记录不摘录未经核实的硬件参数。后续引用具体参数时，应读取相应正文或结合实际设备信息确认。
