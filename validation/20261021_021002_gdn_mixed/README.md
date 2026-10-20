# GDN mixed：修复后正确性、微基准与四组服务对照

**`7aeaa5b46` 的37项正确性测试全部通过；随后按顺序完成微基准和四组服务对照。** 去掉 `torch.profiler` 后，服务内30层 GDN 调用的 host 时间中位数仍约29 ms。`direct_out` 省掉了复制阶段的大部分时间，但本轮没有建立稳定净 host 收益。

服务端固定为 `da572cddb38383deae9caef33b8d490e292ef071`，工具固定为 `7aeaa5b46ec1cdbdb7e5d43b396b4d05af87fa37` 的NVMe快照。GPU为物理1、A100 80GB PCIe；PyTorch 2.13.0+cu129、vLLM 0.32.0+cu129、驱动550.127.05，非官方cu130环境。没有修改用户计算代码、容差、系统驱动、CUDA、频率、功耗或风扇。

此前失败的 `daac4cf10` 运行保留在 [旧门禁结果](../20261021_014640_gdn_mixed/README.md)，没有覆盖。

## 1. 正确性门禁

先单独运行指纹测试，确认通过后才运行完整测试；完整结果 **37 passed / 0 failed / 0 skipped**，退出码0，约21.19秒。

| 检查 | 通过 |
| --- | ---: |
| 源码指纹 | 1 |
| baseline/direct_out 对 installed 的输出、卷积池和SSM池精确相等 | 20 |
| 与标准路径按原容差比较 | 10 |
| 非连续输出回退 | 2 |
| fresh→continued 两步续写 | 4 |

BF16和FP32状态均覆盖，保留行0与未使用行检查通过。当前与预期函数指纹一致：

```text
9af0a98c23727bea4e7f34c1cf3283d838ad7a84a741b2fa8f946d823a4292a9
```

[完整pytest输出](step1/full_tests.log) · [JUnit](step1/full_tests.junit.xml) · [指纹输出](step1/fingerprint.log) · [逐项结果](step1/summary.json)

## 2. 单层微基准

按指定命令运行 C8/C16/C64、BF16状态、200次迭代、CPU `0–27`、`--kernel-time`。指纹输出为 `matches`，退出码0。

| Case | installed host μs p50 | baseline host μs p50 | direct_out host μs p50 | installed host ×30 |
| --- | ---: | ---: | ---: | ---: |
| C8：7 decode +170 prefill tokens | 888.265 | 888.571 | 890.051 | 26.65 ms |
| C16：15 +176 | 998.183 | 1000.678 | 996.779 | 29.95 ms |
| C64：60 +358 | 1002.788 | 1003.899 | 1000.598 | 30.08 ms |

baseline 对 installed 的 host 中位数差异为 **+0.034% / +0.250% / +0.111%**，未见明显的函数拷贝额外开销。direct_out 对 baseline 为 **+0.167% / −0.390% / −0.329%**，方向不一致，不能作为稳定加速证据。

GPU event span约907–1025 μs，与host时间接近；单独CUDA profiler设备时间估计约127–232 μs。这提示该微基准调用可能受host发射制约，但三者不是同一个时间量。`kernel_us` 来自独立profiler pass减去独立restore pass的估计，不能当成无扰动的精确设备时长；host×30也不是实测端到端收益。

[原始微基准JSON](step2/gdn_bench_bf16.json) · [完整输出及kernel/分阶段表](step2/bench.log) · [独立审计](independent-audit/step2_cpu_audit.json)

## 3. 四组服务对照

每组全新服务、C16前256条codes、8条预热排除、不显式传seed、不保存音频。服务CPU `0–27,32–59`、客户端`60–63`，与B1相同。选择用户允许的256条短测方式，**本轮没有启用thermal gate，也没有开启torch.profiler**。

三组probe均核实实际stage0 PID的成功patch，未出现 `NOT patched`。installed组核实无probe环境、无patch日志、无probe记录，不能把它的GDN时间记为0。

| 对照 | 完整probe步数 | GDN host p50 ms | GDN host mean ms | TTFA p50/p95 ms | RTF mean | SM中位数 MHz | 热降频采样 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| baseline，分阶段计时 | 220 | 28.969 | 29.075 | 200.0 / 363.3 | 0.4690 | 1395 | 0% |
| baseline，`LAPS=0` | 219 | 29.093 | 29.581 | 202.6 / 355.6 | 0.4739 | 1380 | 0% |
| installed，无探针 | — | — | — | 204.8 / 360.2 | 0.4833 | 1305 | 50.00% |
| direct_out，分阶段计时 | 217 | 28.822 | 29.217 | 209.0 / 360.8 | 0.4899 | 1290 | 57.14% |

**1024条主请求全部成功。** 每组之后另有33条C1请求，四组总计132条也全部成功；它们仅用于确保探针批量缓存写出，单独存放且不计主测试指标。三组probe的有效记录均为每步30层，逐步分段和等于调用时间之和，且已在关闭服务前证明存在主窗口之后的持久化记录，因此没有依赖 `atexit` 猜测尾部完整。

官方汇总原样保留：

- [baseline分阶段表](service_controls/baseline_laps/official_probe_summary.txt) / [JSON](service_controls/baseline_laps/official_probe_summary.json)
- [baseline LAPS=0](service_controls/baseline_no_laps/official_probe_summary.txt) / [JSON](service_controls/baseline_no_laps/official_probe_summary.json)
- [direct_out分阶段表](service_controls/direct_out_laps/official_probe_summary.txt) / [JSON](service_controls/direct_out_laps/official_probe_summary.json)
- [installed端到端summary](service_controls/installed/client/summary.json)

## 分阶段结果与解读

以下是每个混合步30次GDN调用的 **host阶段均值，单位ms**，不是GPU kernel时间：

| 对照 | setup | conv | decode | prep | gather | chunk | scatter | copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.436 | 3.704 | 2.694 | 2.711 | 2.295 | 15.204 | 1.475 | 0.557 |
| direct_out | 0.449 | 3.713 | 2.703 | 2.761 | 2.316 | 15.752 | 1.496 | 0.026 |

- **GDN仍是几十毫秒量级。** 无profiler测得约29 ms，而之前profile约38–40 ms；不支持把原有现象全部归为profiler开销，但不同运行不能用于精确计算profiler增加了多少时间。
- `chunk` 是最大已计时阶段，包含FLA Python/Triton发射，也包含direct_out的输出视图和参数准备；不能把这个区间增加解释成FLA GPU kernel变慢。
- copy均值减少约0.531 ms，而chunk区间增加约0.548 ms。GDN总host中位数下降0.509%，均值却上升0.489%，**没有清晰、一致的净收益**。
- LAPS=0与分段组接近，但只有一次运行，不能证明打点成本严格为0。
- 逐步GDN sum/span中位数约89.6%–90.8%；span仅为首个GDN调用开始到最后一个GDN调用结束之间的host跨度，不是完整混合步墙钟时间或GPU时间。

## 对照限制与审计

installed和direct_out组出现热降频，四组温频没有匹配，因此不能把端到端TTFA/RTF差异单独归因于探针或候选改动。三probe组共有70个 `(num_prefills, prefill_tokens, decode_tokens)` 形状，但57个形状至少一组不足3个样本；同形状也不代表同请求或同热条件。不作统计显著性或稳定收益断言。

[独立最终结论](independent-audit/FINAL_FINDINGS.md) · [完整证据JSON](independent-audit/FINAL_FINDINGS.json) · [四组温频及形状对照](independent-audit/SERVICE_CONTROLS_COMPARISON.md) · [共同形状原始表](independent-audit/common_shape_host.csv)

用户工具和计算代码保持原字节。外置runner在首组后增加了外来GPU阻断检查和自动导出原官方汇总，旧版、diff说明和各组实际hash均保留，不能称所有管理helper字节完全相同；请求构造和计算条件未因此改变。

本轮未保存音频，未额外进行WER/SIM评测，不能用HTTP成功代替音频质量证明。没有提交上游PR或在服务源码中实施候选优化。

## 清理与复现

[最终清理](final_cleanup.json)确认GPU1为15 MiB / 0%，无本轮API/stage/MPS/monitor进程，8091空闲；16个工具快照文件SHA匹配，服务端tracked源码干净。各阶段命令、环境、probe JSONL、主请求与drain记录、逐秒GPU/HBM遥测均在本目录。

公开副本只替换个人绝对路径、主机名和GPU UUID，服务器日志脱敏后gzip压缩，指标不变。原始数据保留在个人NVMe；[发布清单](publication.json)及根目录SHA256SUMS提供校验。
