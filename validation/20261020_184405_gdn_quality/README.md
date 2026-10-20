# A100 GDN 状态精度质量对照

**两组全量实验及评测完成。本轮未观察到 FP32 状态带来的质量改善，尚不能把 BF16 状态精度认定为异常根因。** 运行日期：2026-10-20。

## 固定条件

- 服务端提交 `da572cddb38383deae9caef33b8d490e292ef071`，模型 `fishaudio/drama-3` revision `b2b5ee998034b2975720906c815e7fb0e9f007bd`。
- 生成、音频分析及 WER/SIM 工具固定于 `4d87d0f35b77b5d14dd5b381d2fde18f5cad26ab`。运行期间新增的 `060ae50` 失败统计采用独立 CPU 后处理，复用原音频、ASR、WER、SIM，原始 `quality.json` 保留。
- A100 80GB PCIe，物理 GPU 1；驱动 550.127.05，PyTorch 2.13.0+cu129，vLLM 0.32.0+cu129。未修改系统驱动或 CUDA。**不是官方 cu130 / R580+ 环境。**
- Seed-TTS-Eval 英文全量 1088 条，codes 模式，C16，`--seed 0 --save-audio`，每组重新启动服务、8 条预热。客户端 seed 规则为 `seed_base + row_index`，两组实际 seed 均为 0–1087。
- [默认 YAML](default.yaml) 和 [FP32 YAML](fp32_state.yaml) 仅一处不同：stage 0 的 `mamba_ssm_cache_dtype: bfloat16` → `float32`。其余配置、输入及运行环境一致，见 [配对完整性](paired_integrity_checks.json)。
- 两组共 2176 个 WAV 全部保存在个人 NVMe。此处上传逐条指标、音频清单、评测结果及复现证据，未上传全量 WAV。不是正式 B1 性能基线。

## GPU 单测

原始测试收集 42 项，精确排除 6 项不适用 A100 的 FlashInfer 用例，实际 **32 passed / 4 failed**。融合采样分布测试通过，sampling、candidates 和 16 项 DAC 测试全部通过。

四个失败是带 `triton` 参数的 GDN 数值对照用例：夹具实际仍绑定 FlashInfer callable，在参考 prefill 阶段报 `GDN prefill DSL kernel is unavailable`，尚未执行数值断言。不能将其解释为 Triton 数值不一致。

仓库外 [补充 plugin](unit_tests/actual_triton_plugin.py) 只把该夹具的 callable 换成安装版生产 `ChunkGatedDeltaRule.forward_native`，保持原输入、seed、断言和容差。BF16/FP32 状态 × 0/3 decode 的 **4 项真实 FLA/Triton 对照全部通过**。这是补充验证，不能把原始四项失败改记为通过。最初 plugin 文件匹配未生效的尝试也保留在证据中。

详见 [单测报告](unit_tests/report.txt)、[结构化结果](unit_tests/summary.json)、[原始日志](unit_tests/pytest.log) 和 [补充日志](unit_tests/supplemental-pytest.log)。小规模单测不能排除长序列生成缺陷。

## 质量结果

| 指标 | 默认 BF16 | FP32 状态 |
| --- | ---: | ---: |
| 成功请求 / WER / SIM 覆盖 | 1088 / 1088 / 1088 | 1088 / 1088 / 1088 |
| 请求失败 / 评测错误 / 空 ASR | 0 / 0 / 0 | 0 / 0 / 0 |
| noise-run | 13（1.19%） | 16（1.47%） |
| 达到 500 帧上限 | 4 | 4 |
| 全量平均 WER | 2.32% | 2.34% |
| 去 noise-run WER | 1.66% | 1.76% |
| 全量平均 SIM（代理） | 0.8646 | 0.8637 |
| 去 noise-run SIM（代理） | 0.8703 | 0.8701 |

两组均完整生成 1088 条且请求失败为 0；每条均有音频、ASR/WER 与 SIM。WER 是逐句 WER 的算术平均，不是将全部词拼接后的 micro WER。Whisper 使用固定 revision 的 `openai/whisper-large-v3`、FP32 和仓库 Seed-TTS 归一化方法。

SIM 是 `microsoft/wavlm-base-plus` 均值池化嵌入余弦代理指标，可在本次两组之间比较，**不能直接与论文中的微调 WavLM-large speaker-verification SIM 比较**。模型 revision 与下载哈希见 [模型准备记录](eval-models/READY.json)；其中模型准备阶段的 `model_weights_loaded: false` 描述下载预检时状态，实际完整评测见 [评测完整性](quality_completeness.json)。

noise-run 按分析脚本的固定阈值自动标记，是疑似停止正常说话的指标，并非每条都已人工听检。默认组 13/1088（1.195%），FP32 组 16/1088（1.471%）；只有 1 条两组共同命中，默认组独有 12 条、FP32 独有 15 条。精确双侧 McNemar p=0.7011；单组 Wilson 95% 区间分别为 0.700%–2.034%、0.907%–2.375%。相同 seed 不保证不同在线调度或精度下逐条生成完全相同，本轮不能证明根因或两种精度等价。

默认 / FP32 组 noise-run 样本的平均 WER 分别为 56.81% / 41.04%；全部 WER>50% 的样本（8 / 7 条）均落在 noise-run 内，说明这批异常关联内容缺失或错误。全量平均 WER 只相差约 0.02 个百分点，SIM 相差约 0.0009；单次对照不支持 FP32 改善质量，也不足以断言 FP32 更差。建议保留 BF16 默认配置作为后续对照，继续定位异常生成，暂不把改精度当成修复。

逐句数据：[默认 quality.json](default/quality.json)、[FP32 quality.json](fp32_state/quality.json)、[配对摘要](quality_pair_summary.json)、[noise-run 与 WER/SIM 关联](noise_quality_association.json)。保留两组全部样本计算主要指标，去 noise-run 的指标仅为诊断，不替代全量指标。

## 新增失败统计口径

| 指标 | 默认 BF16 | FP32 状态 |
| --- | ---: | ---: |
| 音频 ≥20 秒（脚本 #8933 口径） | 6（0.55%） | 8（0.74%） |
| 综合失败（脚本 #9157 口径） | 10（0.92%） | 11（1.01%） |
| 连续静音 >1 秒（脚本 #9044 口径） | 16（1.47%） | 24（2.21%） |

这里严格执行 `060ae50` 脚本的口径：≥20 秒；达到 500 帧或 WER>50% 或时长相对本组每词秒数中位数的预测值超出 0.3–3 倍；以及 50 ms 窗口 RMS<1e-3 连续超过 1 秒。口径来自脚本，未将其视为统一的人工听感标准。

见 [后处理汇总](upstream_failure_summary.json) 和两组 [default](default/quality_upstream.json) / [fp32_state](fp32_state/quality_upstream.json) 逐条结果；[后处理代码](supplement_upstream_counts.py) 校验原有每条字段与 totals 均未改变，无需重复 ASR/SIM。

## 运行与复现证据

| 指标 | 默认 BF16 | FP32 状态 |
| --- | ---: | ---: |
| 服务启动（秒） | 235.352 | 235.101 |
| 生成批次时长（秒） | 169.68 | 215.996 |
| loop-lag p99（ms） | 0.94 | 0.91 |

这些时长来自保存音频的单次质量运行，不作正式性能优化收益结论。FP32 状态配置的 KV 容量从 326,716 降至 317,667 tokens；日志中的 39.88x / 38.78x 是按 8192 tokens/request 估算的并发度，不能直接证明 C128 无抢占。

两次生成日志均未检出 Traceback、ERROR 或抢占记录。关停时服务内部有强制清理剩余 stage 0 子进程的警告，最终服务退出码为 0，端口及相关进程释放；见 [日志审计](server_log_audit.json)。这次 C16 运行并未验证 C128。

- [实验元数据](experiment.json)、[源文件哈希](source_hashes.json)、[运行时使用的工具快照](evaluated-tools-4d87d0f/provenance.json)。
- [默认请求指标](default/summary.json) / [FP32 请求指标](fp32_state/summary.json)。
- [默认后端与 KV 日志](default/key_lines.txt) / [FP32 后端与 KV 日志](fp32_state/key_lines.txt)。
- [默认清理记录](default/cleanup.json) / [FP32 清理记录](fp32_state/cleanup.json)。
- [评测日志](eval_quality.log)、[评测完整性](quality_completeness.json)、[发布清单](publication.json)。

公开副本将个人绝对路径、主机名和 GPU UUID 替换为占位符，指标未改动。`source_sha256` 标识本地原始文件，根目录 `SHA256SUMS` 校验公开副本。全部原始日志、音频及未脱敏证据仍保存在个人目录。

最终评测退出码为 0，GPU 1 回到 15 MiB / 0%，无本轮计算进程且 8091 端口空闲，见 [评测清理记录](eval_cleanup.json)。B1 未启动。
