# Drama 3 / A100 冒烟测试结果

本分支保存 2026-10-20 的 9 条模型生成音频，供人工试听。所有音频为单声道、16-bit PCM WAV。

**状态：冒烟功能检查通过，用户初步听检通过。后续官方 E2E 为 5 passed。这些运行均不是正式性能基线。**

正式实验用的预编码脚本和压测客户端（按 #8933 的测法）见 [tools/drama3_bench/](tools/drama3_bench/README.md)。

## 测试来源

- 模型：`fishaudio/drama-3`，revision `b2b5ee998034b2975720906c815e7fb0e9f007bd`
- vLLM-Omni：[da572cddb38383deae9caef33b8d490e292ef071](https://github.com/advanced233/vllm-omni/commit/da572cddb38383deae9caef33b8d490e292ef071)
- GPU：NVIDIA A100 80GB PCIe
- 实际环境：vLLM 0.32.0+cu129、PyTorch 2.13.0+cu129、CUDA runtime 12.9、驱动 550.127.05；不满足官方 cu130 / R580+ 环境要求。
- 9 条正常请求均返回 HTTP 200；4 路并发全部成功；缺少 ref_text 的请求返回 HTTP 400。
- 采样率 44100 Hz；流式接口返回原始 PCM，客户端按 44100 Hz 协议假设封装 WAV。
- 原始请求指标见 [summary.json](summary.json)。TTFA、RTF 仅作本次记录。

## 工具实机验证

已完成[两种参考方式的C4 × 32真实模型验证](validation/20261020_165001_49599d4/README.md)：数值检查通过，音色及长尾差异听检待完成，尚未开始正式基线。

## 结果与证据

本页音频及根目录指标来自冒烟运行 `20261020_043135_da572cdd`。

- 启动耗时：[startup_seconds.txt](startup_seconds.txt)，272 秒，复用了编译缓存。
- 启动后端、MPS 与 GPU 状态：[key_lines.txt](key_lines.txt)。
- 请求指标：[summary.json](summary.json)；完整说明：[report.txt](report.txt)。
- 原始自动检查：[automated_validation.json](automated_validation.json)。其 `duration_review_needed` 是听检前的历史状态；后续反馈单独记录在 [listening_review.json](listening_review.json)。
- 运行元数据：[manifest.json](manifest.json)；部署配置：[deploy_config.yaml](deploy_config.yaml)；客户端：[smoke_client.py](smoke_client.py)。
- GPU 逐秒记录：[gpu.csv](gpu.csv)；客户端输出：[client.log](client.log)。

用户反馈：00、01 是男声，其余克隆输出为女声且音色相近，整体没有明显问题。

## 后续官方端到端测试

独立运行 `20261020_125552_nvme_da572cdd` 在权重迁至 NVMe 后执行：

```bash
python -m pytest tests/e2e/online_serving/test_drama_3.py -v -s --run-level advanced_model
```

- **5 passed，0 errors / failures / skipped，退出码 0**。
- 7 条成功音频均完成 Whisper small 内容校验，相似度均为 1.000。
- 两条流式 HNR 为 1.54、1.65 dB，门槛为 −1 dB；4 路并发全部成功；缺少 `ref_text` 正确返回 400。
- 服务启动 344.925 秒，stage 0 权重加载 38.25 秒。完整权重哈希校验刚完成，不能将其视为受控冷启动。
- 关停出现内部 force-kill / semaphore 清理警告，最终进程、MPS 和端口均释放，GPU 1 回到 15 MiB / 0%。
- 官方 helper 自动清理了 E2E 临时 WAV，因此本分支的 9 个 WAV 仍全部来自前面的冒烟运行。

详见 [E2E 报告](e2e/report.txt)、[E2E summary](e2e/summary.json)、[E2E 关键日志](e2e/key_lines.txt)。此前 HDD 上的一次 E2E 启动超时记录保留在 [previous_attempt.json](e2e/previous_attempt.json)；该次没有执行音频请求。

## 发布副本说明

本机绝对路径替换为 `${WORKSPACE}`、`${PERSONAL_WORKDIR}`、`${USER_HOME}`、`${NVME_WORKDIR}`；GPU UUID 统一替换为 `GPU_PHYSICAL_1`，无关 GPU 的身份行省略。日志行号仍指向本地原始日志。原始音频、客户端请求指标和 GPU 采样值未改动。报告中提及的其他本地辅助文件未全部上传；公开证据以本页链接的文件为准。

## 试听文本

`00_warmup_text`、`01_text`：

> The weather is nice today, perfect for a walk in the park.

其余 7 条克隆音频：

> The quarterly report is due on Thursday, so please finish it by noon.

声音克隆参考音频为仓库自带的 [clone_2.wav](https://github.com/advanced233/vllm-omni/blob/da572cddb38383deae9caef33b8d490e292ef071/tests/assets/qwen3_tts/clone_2.wav)。

## 音频文件

点击文件后使用 GitHub 的 Raw / 下载按钮保存到本地试听。也可以通过本分支的 Code → Download ZIP 一次下载。

| 文件 | 时长（秒） | RMS（dBFS） |
| --- | ---: | ---: |
| [00_warmup_text.wav](audio/00_warmup_text.wav) | 3.622 | -16.1 |
| [01_text.wav](audio/01_text.wav) | 4.040 | -16.0 |
| [02_clone.wav](audio/02_clone.wav) | 3.994 | -22.0 |
| [03_clone_stream_seed0.wav](audio/03_clone_stream_seed0.wav) | 3.297 | -19.3 |
| [03_clone_stream_seed42.wav](audio/03_clone_stream_seed42.wav) | 3.901 | -22.4 |
| [04_concurrent_0.wav](audio/04_concurrent_0.wav) | 3.855 | -22.3 |
| [04_concurrent_1.wav](audio/04_concurrent_1.wav) | 3.994 | -23.0 |
| [04_concurrent_2.wav](audio/04_concurrent_2.wav) | 4.180 | -22.0 |
| [04_concurrent_3.wav](audio/04_concurrent_3.wav) | 3.669 | -22.2 |

## 听检要点

建议先听最短的 `03_clone_stream_seed0.wav`，再与 `02_clone.wav`、seed 42 和参考音频比较，最后检查 4 条并发输出。

- 对照请求文本确认整句完整、清晰，无漏词、重复或多余内容。
- 检查开头和结尾是否截断，是否出现跳音、异常停顿、爆音或明显失真。
- 克隆音色应与参考大致一致，途中不应突然换人。
- 4 条并发输出均应独立完整。

克隆时长约 4–7 秒只是参考，短于 4 秒本身不代表失败。用户后续反馈整体听起来没有明显问题；3.994 秒可视为约 4 秒。此反馈是初步听检，不是标准化主观评分。

文件校验值见 [SHA256SUMS](SHA256SUMS)。
