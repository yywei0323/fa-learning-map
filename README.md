# FA 算子开发学习地图

> 面向从零开始学习 **FlashAttention（FA）与昇腾 Ascend C 算子开发** 的学习仓库。

## 先说结论

FA 不是“把 Attention 换成一个近似算法”，而是在保持 Attention 数学含义（允许浮点误差）的前提下，通过：

1. 分块计算（Tiling）
2. Online Softmax
3. 减少中间矩阵写回全局内存
4. 让数据搬运与计算并行

降低内存访问开销。FA 算子开发的核心不是把公式翻译成 C++，而是同时处理：

- 数学正确性
- 数值稳定性
- NPU 存储层次与计算单元
- Shape、Layout、Mask、数据类型
- Tiling、并行、流水与性能验证

## 学习入口

1. [00｜FA 零基础扫盲](docs/00-fa-primer.md)
2. [学习路线图](docs/roadmap.md)
3. [术语表](docs/glossary.md)

## 仓库规划

```text
fa-learning-map/
├── README.md
├── docs/
│   ├── 00-fa-primer.md
│   ├── roadmap.md
│   └── glossary.md
├── notes/          # 后续：逐章学习笔记
├── examples/       # 后续：可运行的最小实验
└── benchmarks/     # 后续：正确性和性能记录
```

## 当前学习原则

- 先理解普通 Attention，再理解 FlashAttention。
- 先用小矩阵手算和 Python 参考实现验证，再写 Ascend C Kernel。
- 先正确，再优化；每次优化都用数据证明。
- 不把“FA 算法”“FA 算子接口”“某个硬件上的 FA 实现”混为一谈。

## 官方资料

- [Ascend C 自定义算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)
- [Ascend C FlashAttention 性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240607-1)
- [Ascend C Add 算子快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0005.html)

> 文档版本、支持芯片和 API 会随 CANN 版本变化。真正开始搭环境时，应固定芯片型号、CANN 与 torch_npu 版本。
