# Drama 3 / A100：decode 与混合步 profile

**C8、C16、C64 均已完成真实双阶段 MPS profile，并通过独立 CUDA graph 节点、关联和步骤边界审计。** 这批诊断运行不计入 [17 次 B1 基线](../20261020_194158_baseline/README.md)。尚未实施优化，也没有测得或宣称优化收益。

下一项建议：先在关闭 `torch.profiler` 的情况下，用低开销计时验证 **GDN/FLA 混合步 CPU 路径**；如果现象仍成立，再隔离测试已有输出缓冲直写。GPU 侧明确的主线是 **MoE/Marlin**，需要在 vLLM 依赖侧用实际专家形状做微基准。详见 [最终建议与源码证据](source-analysis/FINAL_RECOMMENDATIONS.txt)。

## 采集与验收

- 服务端 `da572cddb38383deae9caef33b8d490e292ef071`，压测工具 `3b116ea1a7f39323e517804900aa7bb016bafdf1`，A100 80GB PCIe、cu129、驱动550.127.05。未修改模型计算源码、系统驱动或 CUDA。
- 每档新服务，默认 BF16 部署仅增加两阶段 profiler 配置；同 B1 的 CPU 亲和性、codes、1088条、8条预热、默认采样、不显式传 seed、不保存音频。
- 计时负载开始20秒后调用两阶段 `/start_profile`，约5秒后 `/stop_profile`；stage 0 最多256步、stage 1 最多4096步，谁先到限谁先停。使用各阶段实际 capture 窗口，而非假设它们完全同步。
- 最终使用 **C8 attempt2、C16 attempt1、C64 attempt1**。最初 C8 attempt1 的 stage 1 上限为256，把空轮询也计入，导致共同窗口仅约0.447秒；该次保留作诊断记录，没有与重采混算。
- 真实小图探针在 profiler 启动前捕获图，3次回放各有2个内部 kernel，实际 correlation 均正确。解析时排除 PyTorch 复制的 `gpu_user_annotation`，只计 CPU 原生执行 scope。
- 独立分析保守剔除每线程首尾及不完整步骤，按真实 scheduler context/generation 标记分类，再通过 CUPTI correlation 关联 kernel；不靠 kernel 名猜测 prefill/decode。
- 三档最终 trace 均无重复 kernel、无 correlation 冲突，graph launch 都有内部节点；未归因 kernel 时间占比约0.103% / 0.085% / 0.058%，没有强行分配这些活动。

四次模型 profile（含 C8 初次诊断）各1088条全部成功。最终 GPU 1 为15 MiB / 0%，8091空闲，API/stage/MPS/采样及压缩进程均退出；见 [最终清理](final_cleanup.json)。

## 主要观测

| 客户端并发 | 完整 decode / mixed 步 | decode GPU span p50 | mixed GPU span p50 | decode Marlin 占比 | mixed GDN CPU scope union p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 232 / 22 | 13.744 ms | 49.067 ms | 30.94% | 40.011 ms |
| 16 | 194 / 29 | 18.063 ms | 46.324 ms | 38.60% | 38.297 ms |
| 64 | 89 / 29 | 33.179 ms | 66.695 ms | 46.12% | 39.611 ms |

**GPU span、kernel 累计时长占比、CPU inclusive scope 是三种不同的量。** Kernel 累计时长可能包含并行重叠；CPU GDN scope 包含嵌套工作与 profiler 开销，不能再加上其内部 FLA scope，也不能当成可节省的38–40 ms。

每步实际 batch 不是固定等于客户端并发。纯 decode 的 generation-request 中位数为8/16/62；mixed 的 context-token 中位数为169.5/176/358，分布也不同。因此不能从 C8 mixed 49 ms、C16 mixed 46 ms推导并发更高反而更快。完整分布、p95和定义见 [独立精炼总表](independent-audit/INDEPENDENT_PROFILE_FINDINGS.md) 与 [JSON](independent-audit/INDEPENDENT_PROFILE_FINDINGS.json)。

源码与计数匹配：

- 每个完整 decode 步有80次 Marlin kernel，对应40层 backbone MoE、每层两次 GEMM；每步30次 packed GDN decode 与模型结构相符。
- 通用 dense GEMM/GEMV 及 BLAS reduction 也占较大份额，但同名 kernel 跨 backbone/tower，不能只靠名字把时间全归给 codebook tower。
- 纯 decode 每步1次 graph launch，mixed 每步54次。真实编译缓存包含54个编译分区与53个 attention 分区；分区会跨模块边界，不能简单称为54个完整层。见 [实际编译分区存证](source-analysis/piecewise_54_evidence.json)。

## 混合步 CPU 开销究竟是什么

三档 mixed CPU execute scope p50 为58.95 / 58.11 / 60.27 ms，但 CUDA runtime API 区间 union 约8 ms，54次 GraphLaunch 的平均合计时间仅约2.7–2.8 ms。未观察到这些 scope 内显式 `cudaStream/Device/EventSynchronize` 或 D2H。

GDN相关同名 CPU scope 去重 union 约38–40 ms，在三个并发点均出现。这个结果支持优先调查 GDN/FLA 的 Python/算子调度路径，**不支持把全部耗时归为图启动、GPU同步等待，或断言这些时间可以直接消除**。需先用不逐层打印、不额外 CUDA synchronize 的低开销计时排除 profiler 扰动。

独立 CPU 证据：[C8](independent-audit/c8-attempt2/cpu_waits_stage0.json)、[C16](independent-audit/c16/cpu_waits_stage0.json)、[C64](independent-audit/c64/cpu_waits_stage0.json)。

## 下一项实验的选择

1. **Omni 内先验证 GDN mixed host 路径与输出缓冲复用。** 当前 FLA prefill 调用未传 `core_attn_out`，随后再复制结果，而安装版调用链已有输出缓冲参数。先验证低开销 CPU 计时，再单独实验直写缓冲，覆盖 BF16/FP32、fresh/continued/chunked prefill、slot reuse 和输出别名。这里只识别了候选，没有改代码或测得收益。见 [具体候选与验证方法](source-analysis/gdn_host_candidate.json)。
2. **GPU 优化主线：真实 A100 MoE/Marlin 形状调优。** 同权重、router indices、激活及规约语义下测试支持的 tile/block 参数，再做匹配热条件的端到端对照。实现位置属于 vLLM 依赖，不能包装成只改 Omni 的性能 PR。
3. **Prefill admission 是有代价的配置实验。** 默认 `min_running=32` 在 C8/C16 下不触发 admission batching；可单独试较低阈值，检查 mixed-step 频率、stall、TTFA p95/p99、吞吐和公平性。等待新请求可能抵消节省，不承诺改善。

采样融合暂列低优先；没有以降低采样候选数、改概率分母或改精度制造性能收益。任何后续变更仍需算子正确性、全量音频质量与端到端验证。源文件行号、SHA、实际形状和限制见 [源码映射](source-analysis/source_mapping.json)。

## 温频与 MPS 的边界

| C | 采集时 SM 中位数 | 温度 | 软件热降频采样 | 两阶段共同窗口 | GPU busy 同时活动 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1410 MHz | 82–83°C | 0/5 | 4.423 s | 393 ms |
| 16 | 1305 MHz | 84°C | 4/5 | 5.036 s | 626 ms |
| 64 | 1155 MHz | 84–85°C | 6/6 | 5.109 s | 1016 ms |

这些是与实际 trace 时间对齐的1秒采样；三档热状态不同，且不等于 B1 长跑的频率分布。两阶段时间锚点、同一主机和实际窗口交集已核验；重叠只证明同时存在 GPU 活动，不能据此计算“MPS拖慢百分比”。没有从 kernel 份额推算 Amdahl 加速比。

## 文件与复核

- 每个 `c*/attempt*/` 包含配置、启动命令、源版本、逐秒硬件记录、API响应、客户端指标、压缩服务器日志、分析和清理记录。
- `independent-audit/` 保留保守步骤选择、raw kernel表、CPU API统计和硬件对齐；`source-analysis/` 提供源码映射。修订过的启发式分类也保留旧版，最终使用已将 Gemma norm 与 GEMM 区分的版本。
- 完整模型 trace 压缩文件留在个人 NVMe，没有上传大 trace；[私有 trace 清单](private_trace_manifest.json) 保存相对位置、大小和SHA256。小图探针保留在 `graph_probe/`。
- 公开文件仅替换个人绝对路径、主机名与 GPU UUID，数值不变，见 [发布清单](publication.json)。原始未脱敏证据仍保留。
