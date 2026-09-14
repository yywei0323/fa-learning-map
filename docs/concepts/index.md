# 全部术语索引

[八层目录](README.md) · [概念关系](relations.md) · [来源与覆盖](sources.md)

共 262 个索引项，别名合并在同一项中；这是导航条目数，不是已掌握概念数量。同一主题在不同层出现时按语境区分。可用页面搜索定位缩写。

## 1. 张量与代码

| 术语 / 别名 | 解释所在页 |
|---|---|
| Tensor | [查看解释](01-foundations/tensors-and-code.md) |
| Scalar | [查看解释](01-foundations/tensors-and-code.md) |
| Rank | [查看解释](01-foundations/tensors-and-code.md) |
| Shape | [查看解释](01-foundations/tensors-and-code.md) |
| dtype | [查看解释](01-foundations/tensors-and-code.md) |
| device | [查看解释](01-foundations/tensors-and-code.md) |
| numel | [查看解释](01-foundations/tensors-and-code.md) |
| stride | [查看解释](01-foundations/tensors-and-code.md) |
| Contiguous | [查看解释](01-foundations/tensors-and-code.md) |
| Index / Slice | [查看解释](01-foundations/tensors-and-code.md) |
| Broadcasting | [查看解释](01-foundations/tensors-and-code.md) |
| view / reshape | [查看解释](01-foundations/tensors-and-code.md) |
| transpose / permute | [查看解释](01-foundations/tensors-and-code.md) |
| unsqueeze / squeeze | [查看解释](01-foundations/tensors-and-code.md) |
| matmul / @ | [查看解释](01-foundations/tensors-and-code.md) |
| Elementwise | [查看解释](01-foundations/tensors-and-code.md) |
| Reduce | [查看解释](01-foundations/tensors-and-code.md) |
| Cast | [查看解释](01-foundations/tensors-and-code.md) |
| nn.Module / forward / __init__ | [查看解释](01-foundations/tensors-and-code.md) |
| nn.Parameter | [查看解释](01-foundations/tensors-and-code.md) |
| register_buffer | [查看解释](01-foundations/tensors-and-code.md) |
| ModuleList / ModuleDict | [查看解释](01-foundations/tensors-and-code.md) |
| nn.Embedding | [查看解释](01-foundations/tensors-and-code.md) |
| nn.Linear | [查看解释](01-foundations/tensors-and-code.md) |
| Autograd / backward / grad | [查看解释](01-foundations/tensors-and-code.md) |
| train / eval | [查看解释](01-foundations/tensors-and-code.md) |
| no_grad / inference_mode | [查看解释](01-foundations/tensors-and-code.md) |
| state_dict | [查看解释](01-foundations/tensors-and-code.md) |
| Compile time / Runtime | [查看解释](01-foundations/tensors-and-code.md) |
| Template | [查看解释](01-foundations/tensors-and-code.md) |
| if constexpr | [查看解释](01-foundations/tensors-and-code.md) |
| struct / enum | [查看解释](01-foundations/tensors-and-code.md) |
| Pointer / Offset | [查看解释](01-foundations/tensors-and-code.md) |
| Namespace / Header / Macro | [查看解释](01-foundations/tensors-and-code.md) |
| CMake | [查看解释](01-foundations/tensors-and-code.md) |
| Git Commit / Branch / PR | [查看解释](01-foundations/tensors-and-code.md) |
| Notebook | [查看解释](01-foundations/tensors-and-code.md) |

## 2. 模型与推理

| 术语 / 别名 | 解释所在页 |
|---|---|
| LLM / 语言模型 | [查看解释](02-model/transformer-and-inference.md) |
| Token / Tokenizer | [查看解释](02-model/transformer-and-inference.md) |
| Vocabulary / Token ID | [查看解释](02-model/transformer-and-inference.md) |
| Embedding | [查看解释](02-model/transformer-and-inference.md) |
| Hidden State / Activation | [查看解释](02-model/transformer-and-inference.md) |
| Parameter / Weight | [查看解释](02-model/transformer-and-inference.md) |
| Context length | [查看解释](02-model/transformer-and-inference.md) |
| Batch / 并发 | [查看解释](02-model/transformer-and-inference.md) |
| Encoder | [查看解释](02-model/transformer-and-inference.md) |
| Decoder | [查看解释](02-model/transformer-and-inference.md) |
| Decoder-only | [查看解释](02-model/transformer-and-inference.md) |
| Self-Attention | [查看解释](02-model/transformer-and-inference.md) |
| Cross-Attention | [查看解释](02-model/transformer-and-inference.md) |
| Q/K/V Projection / Wo | [查看解释](02-model/transformer-and-inference.md) |
| 拆头 / 合头 | [查看解释](02-model/transformer-and-inference.md) |
| Residual | [查看解释](02-model/transformer-and-inference.md) |
| Pre-Norm / Post-Norm | [查看解释](02-model/transformer-and-inference.md) |
| LayerNorm | [查看解释](02-model/transformer-and-inference.md) |
| RMSNorm | [查看解释](02-model/transformer-and-inference.md) |
| MLP / FFN | [查看解释](02-model/transformer-and-inference.md) |
| GELU / SiLU / ReLU | [查看解释](02-model/transformer-and-inference.md) |
| MoE | [查看解释](02-model/transformer-and-inference.md) |
| Positional Encoding | [查看解释](02-model/transformer-and-inference.md) |
| RoPE | [查看解释](02-model/transformer-and-inference.md) |
| Dropout | [查看解释](02-model/transformer-and-inference.md) |
| LM Head | [查看解释](02-model/transformer-and-inference.md) |
| Logits | [查看解释](02-model/transformer-and-inference.md) |
| Greedy / Sampling | [查看解释](02-model/transformer-and-inference.md) |
| Temperature / Top-k / Top-p | [查看解释](02-model/transformer-and-inference.md) |
| BOS / EOS / PAD | [查看解释](02-model/transformer-and-inference.md) |
| Teacher forcing / 右移标签 | [查看解释](02-model/transformer-and-inference.md) |
| Cross-entropy / Loss | [查看解释](02-model/transformer-and-inference.md) |
| Backpropagation / Optimizer | [查看解释](02-model/transformer-and-inference.md) |
| Training / Inference | [查看解释](02-model/transformer-and-inference.md) |
| Prefill | [查看解释](02-model/transformer-and-inference.md) |
| Decode | [查看解释](02-model/transformer-and-inference.md) |
| KV Cache | [查看解释](02-model/transformer-and-inference.md) |
| Dynamic / Static / Offloaded / Quantized Cache | [查看解释](02-model/transformer-and-inference.md) |

## 3. Attention 数学

| 术语 / 别名 | 解释所在页 |
|---|---|
| Attention | [查看解释](03-attention/attention-and-fa.md) |
| Q / K / V | [查看解释](03-attention/attention-and-fa.md) |
| B / N1 / N2 / H / D / Dv | [查看解释](03-attention/attention-and-fa.md) |
| S1 / Sq / S2 / Skv | [查看解释](03-attention/attention-and-fa.md) |
| Score / Attention Logit | [查看解释](03-attention/attention-and-fa.md) |
| BMM1 / MM1 | [查看解释](03-attention/attention-and-fa.md) |
| softmax_scale | [查看解释](03-attention/attention-and-fa.md) |
| Mask | [查看解释](03-attention/attention-and-fa.md) |
| Causal Mask | [查看解释](03-attention/attention-and-fa.md) |
| Padding Mask | [查看解释](03-attention/attention-and-fa.md) |
| PSE | [查看解释](03-attention/attention-and-fa.md) |
| Softmax | [查看解释](03-attention/attention-and-fa.md) |
| BMM2 / MM2 / PV | [查看解释](03-attention/attention-and-fa.md) |
| Attention Output | [查看解释](03-attention/attention-and-fa.md) |
| MHA | [查看解释](03-attention/attention-and-fa.md) |
| GQA / G | [查看解释](03-attention/attention-and-fa.md) |
| MQA | [查看解释](03-attention/attention-and-fa.md) |
| FA / FlashAttention | [查看解释](03-attention/attention-and-fa.md) |
| IO-aware | [查看解释](03-attention/attention-and-fa.md) |
| Fusion | [查看解释](03-attention/attention-and-fa.md) |
| Recompute | [查看解释](03-attention/attention-and-fa.md) |
| FA2 | [查看解释](03-attention/attention-and-fa.md) |
| Online Softmax | [查看解释](03-attention/online-softmax.md) |
| 行最大值 m | [查看解释](03-attention/online-softmax.md) |
| 累计指数和 l | [查看解释](03-attention/online-softmax.md) |
| 未归一化输出 u / acc | [查看解释](03-attention/online-softmax.md) |
| expMax / emax / alpha | [查看解释](03-attention/online-softmax.md) |
| 独立块统计合并 | [查看解释](03-attention/online-softmax.md) |
| 无效行 / 全 Mask 行 | [查看解释](03-attention/online-softmax.md) |
| LSE / LogSumExp | [查看解释](03-attention/LSE.md) |

## 4. 布局与寻址

| 术语 / 别名 | 解释所在页 |
|---|---|
| Layout | [查看解释](04-layout/layout.md) |
| ND / FORMAT_ND | [查看解释](04-layout/layout.md) |
| NZ / 分块物理格式 | [查看解释](04-layout/layout.md) |
| BSH | [查看解释](04-layout/layout.md) |
| NTD | [查看解释](04-layout/layout.md) |
| BNSD | [查看解释](04-layout/BNSD.md) |
| BSND | [查看解释](04-layout/BSND.md) |
| TND | [查看解释](04-layout/TND.md) |
| T / 总有效 Token 数 | [查看解释](04-layout/TND.md) |
| Packed / 变长打包 | [查看解释](04-layout/TND.md) |
| cu_seqlens | [查看解释](04-layout/TND.md) |
| Padding / 有效长度 | [查看解释](04-layout/TND.md) |

## 5. 硬件与流水

| 术语 / 别名 | 解释所在页 |
|---|---|
| AI Core | [查看解释](05-hardware/ascend.md) |
| AIC | [查看解释](05-hardware/ascend.md) |
| AIV | [查看解释](05-hardware/ascend.md) |
| Cube | [查看解释](05-hardware/ascend.md) |
| Vector | [查看解释](05-hardware/ascend.md) |
| Scalar 控制单元 | [查看解释](05-hardware/ascend.md) |
| MTE | [查看解释](05-hardware/ascend.md) |
| AI CPU | [查看解释](05-hardware/ascend.md) |
| GM | [查看解释](05-hardware/ascend.md) |
| HBM | [查看解释](05-hardware/ascend.md) |
| L1 Buffer | [查看解释](05-hardware/ascend.md) |
| L0A / L0B / L0C | [查看解释](05-hardware/ascend.md) |
| UB | [查看解释](05-hardware/ascend.md) |
| Vector Register | [查看解释](05-hardware/ascend.md) |
| VF / Vector Function | [查看解释](05-hardware/ascend.md) |
| GlobalTensor | [查看解释](05-hardware/ascend.md) |
| LocalTensor | [查看解释](05-hardware/ascend.md) |
| DataCopy / LoadData | [查看解释](05-hardware/ascend.md) |
| Mmad | [查看解释](05-hardware/ascend.md) |
| Exp / ReduceMax / ReduceSum | [查看解释](05-hardware/ascend.md) |
| Event / Wait / Barrier | [查看解释](05-hardware/ascend.md) |
| Queue / Pipe | [查看解释](05-hardware/ascend.md) |
| Double Buffer / Ping-pong | [查看解释](05-hardware/ascend.md) |
| SoftMax / SimpleSoftMax | [查看解释](05-hardware/ascend.md) |
| SoftmaxFlash / SoftmaxFlashV2 | [查看解释](05-hardware/ascend.md) |

## 6. 算子实现

| 术语 / 别名 | 解释所在页 |
|---|---|
| Operator / 算子 | [查看解释](06-implementation/host-kernel.md) |
| Framework / 接入层 | [查看解释](06-implementation/host-kernel.md) |
| 算子原型 | [查看解释](06-implementation/host-kernel.md) |
| aclnn | [查看解释](06-implementation/host-kernel.md) |
| Host / op_host | [查看解释](06-implementation/host-kernel.md) |
| Device / op_kernel | [查看解释](06-implementation/host-kernel.md) |
| Kernel | [查看解释](06-implementation/host-kernel.md) |
| InferDataType | [查看解释](06-implementation/host-kernel.md) |
| Workspace | [查看解释](06-implementation/host-kernel.md) |
| Stream | [查看解释](06-implementation/host-kernel.md) |
| blockDim / coreNum | [查看解释](06-implementation/host-kernel.md) |
| Metadata | [查看解释](06-implementation/host-kernel.md) |
| Eager / Meta | [查看解释](06-implementation/host-kernel.md) |
| InferShape / Shape Inference | [查看解释](06-implementation/InferShape.md) |
| Tiling / tilling | [查看解释](06-implementation/Tiling.md) |
| Tile | [查看解释](06-implementation/Tiling.md) |
| TilingData | [查看解释](06-implementation/TilingData.md) |
| TilingKey | [查看解释](06-implementation/TilingKey.md) |
| Config / 模板配置 | [查看解释](06-implementation/TilingKey.md) |
| BNS1 分核 | [查看解释](06-implementation/scheduling-and-matmul.md) |
| BN2GS1 分核 | [查看解释](06-implementation/scheduling-and-matmul.md) |
| Query tile / KV tile | [查看解释](06-implementation/scheduling-and-matmul.md) |
| Souter / Sinner | [查看解释](06-implementation/scheduling-and-matmul.md) |
| Alignment | [查看解释](06-implementation/scheduling-and-matmul.md) |
| Tail / 尾块 | [查看解释](06-implementation/scheduling-and-matmul.md) |
| Load balance | [查看解释](06-implementation/scheduling-and-matmul.md) |
| Split-KV / FlashDecode | [查看解释](06-implementation/scheduling-and-matmul.md) |
| M / N / K 矩阵维度 | [查看解释](06-implementation/scheduling-and-matmul.md) |
| MatmulBase | [查看解释](06-implementation/scheduling-and-matmul.md) |
| MatmulK / MatmulN | [查看解释](06-implementation/scheduling-and-matmul.md) |
| MatmulFull | [查看解释](06-implementation/scheduling-and-matmul.md) |
| MxMatmulFull | [查看解释](06-implementation/scheduling-and-matmul.md) |
| ND / DN 函数 | [查看解释](06-implementation/scheduling-and-matmul.md) |

## 7. 推理特性与算子

| 术语 / 别名 | 解释所在页 |
|---|---|
| PA / PagedAttention | [查看解释](07-features/paging-sparsity.md) |
| Page / KV block | [查看解释](07-features/paging-sparsity.md) |
| Logical page / Physical page | [查看解释](07-features/paging-sparsity.md) |
| blockTable / block_table | [查看解释](07-features/paging-sparsity.md) |
| pa_block_size | [查看解释](07-features/paging-sparsity.md) |
| PA_BNBD | [查看解释](07-features/paging-sparsity.md) |
| seqused | [查看解释](07-features/paging-sparsity.md) |
| Sparse / Dense | [查看解释](07-features/paging-sparsity.md) |
| Block Sparse | [查看解释](07-features/paging-sparsity.md) |
| sparse_indices | [查看解释](07-features/paging-sparsity.md) |
| sparse_seq_len | [查看解释](07-features/paging-sparsity.md) |
| Sparse mapping | [查看解释](07-features/paging-sparsity.md) |
| Sparse density | [查看解释](07-features/paging-sparsity.md) |
| Empty sparse row | [查看解释](07-features/paging-sparsity.md) |
| Sink / Window / TopK blocks | [查看解释](07-features/paging-sparsity.md) |
| qflat / kflat | [查看解释](07-features/paging-sparsity.md) |
| vbias | [查看解释](07-features/paging-sparsity.md) |
| Quantization / Dequantization | [查看解释](07-features/quantization.md) |
| INT8 | [查看解释](07-features/quantization.md) |
| FP16 | [查看解释](07-features/quantization.md) |
| BF16 | [查看解释](07-features/quantization.md) |
| FP32 Accumulation | [查看解释](07-features/quantization.md) |
| FP8 E4M3 / E5M2 | [查看解释](07-features/quantization.md) |
| 指数位 / 尾数位 | [查看解释](07-features/quantization.md) |
| ULP | [查看解释](07-features/quantization.md) |
| Rounding / 舍入 | [查看解释](07-features/quantization.md) |
| Clipping / Saturation | [查看解释](07-features/quantization.md) |
| Overflow / Underflow | [查看解释](07-features/quantization.md) |
| Subnormal / 次正规数 | [查看解释](07-features/quantization.md) |
| Scale / Descale | [查看解释](07-features/quantization.md) |
| MXFP8 | [查看解释](07-features/quantization.md) |
| E8M0 | [查看解释](07-features/quantization.md) |
| Quant group | [查看解释](07-features/quantization.md) |
| 数值精度 / 模型精度 | [查看解释](07-features/quantization.md) |
| FIA / FusedInferAttentionScore | [查看解释](07-features/operator-map.md) |
| PFA / PromptFlashAttention | [查看解释](07-features/operator-map.md) |
| IFA / IncreFlashAttention | [查看解释](07-features/operator-map.md) |
| FlashAttentionScore / flash_attn | [查看解释](07-features/operator-map.md) |
| QBSA / QuantBlockSparseAttn | [查看解释](07-features/operator-map.md) |
| QBSA-MXFP8 / quant_mode | [查看解释](07-features/operator-map.md) |
| StemIndexer | [查看解释](07-features/operator-map.md) |
| FIA V3 / V4 / V5 | [查看解释](07-features/operator-map.md) |
| FA1 / FA2 / FA3 | [查看解释](07-features/operator-map.md) |
| MLA | [查看解释](07-features/operator-map.md) |
| q_descale / k_descale / v_descale | [查看解释](07-features/operator-map.md) |
| p_scale | [查看解释](07-features/operator-map.md) |
| per-token-group / per-channel-group | [查看解释](07-features/operator-map.md) |

## 8. 验证与性能

| 术语 / 别名 | 解释所在页 |
|---|---|
| Golden / Reference | [查看解释](08-validation/performance-and-tests.md) |
| UT | [查看解释](08-validation/performance-and-tests.md) |
| ST | [查看解释](08-validation/performance-and-tests.md) |
| TTK | [查看解释](08-validation/performance-and-tests.md) |
| aclnn Fuzz | [查看解释](08-validation/performance-and-tests.md) |
| Ascend C Debug Tool / msDebug | [查看解释](08-validation/performance-and-tests.md) |
| Breakpoint / Call stack | [查看解释](08-validation/performance-and-tests.md) |
| OOB / 越界 | [查看解释](08-validation/performance-and-tests.md) |
| Race / 同步错误 | [查看解释](08-validation/performance-and-tests.md) |
| Regression | [查看解释](08-validation/performance-and-tests.md) |
| Seed / Case | [查看解释](08-validation/performance-and-tests.md) |
| atol / rtol | [查看解释](08-validation/performance-and-tests.md) |
| MaxAbs / RMSE / RelativeL2 | [查看解释](08-validation/performance-and-tests.md) |
| NaN / Inf | [查看解释](08-validation/performance-and-tests.md) |
| Cube bound | [查看解释](08-validation/performance-and-tests.md) |
| Vector bound | [查看解释](08-validation/performance-and-tests.md) |
| Memory / Bandwidth bound | [查看解释](08-validation/performance-and-tests.md) |
| Synchronization bound | [查看解释](08-validation/performance-and-tests.md) |
| Load imbalance | [查看解释](08-validation/performance-and-tests.md) |
| FLOPs / FLOP/s | [查看解释](08-validation/performance-and-tests.md) |
| Bandwidth | [查看解释](08-validation/performance-and-tests.md) |
| Latency | [查看解释](08-validation/performance-and-tests.md) |
| Throughput | [查看解释](08-validation/performance-and-tests.md) |
| Profiling | [查看解释](08-validation/performance-and-tests.md) |
| Pipeline | [查看解释](08-validation/performance-and-tests.md) |
| Warmup / 预热 | [查看解释](08-validation/performance-and-tests.md) |
| Linux / WSL2 | [查看解释](08-validation/performance-and-tests.md) |
| SSH / Remote IDE | [查看解释](08-validation/performance-and-tests.md) |
| Codespaces / Dev Container | [查看解释](08-validation/performance-and-tests.md) |
| Docker / Image | [查看解释](08-validation/performance-and-tests.md) |
| CANN | [查看解释](08-validation/performance-and-tests.md) |
| Ascend C | [查看解释](08-validation/performance-and-tests.md) |
| torch_npu / Driver | [查看解释](08-validation/performance-and-tests.md) |
| clang-format / clang-tidy | [查看解释](08-validation/performance-and-tests.md) |
| CI / Build / Test | [查看解释](08-validation/performance-and-tests.md) |
| 固定 S 与有效 S | [查看解释](08-validation/layout-case.md) |
| querySlot / lseSlot | [查看解释](08-validation/layout-case.md) |
| queryRowStride / queryScaleRowStride | [查看解释](08-validation/layout-case.md) |
| None 与空 Tensor | [查看解释](08-validation/layout-case.md) |
| 布局回归 | [查看解释](08-validation/layout-case.md) |
