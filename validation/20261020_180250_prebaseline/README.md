# B1 前检查 A/B：帧数通过，输出质量异常待查

**检查A通过；检查B的20个HTTP请求均成功，但出现4条低能量长输出，尚不满足“没问题再开始B1”的放行条件。C128短测及13轮B1均未启动。**

工具执行版本 `eaaee46077a352ba7a56a0de5e30ef7049e208c4`，服务源码 `da572cddb38383deae9caef33b8d490e292ef071`。A100 80GB PCIe / cu129 / R550；源代码、采样参数和分析阈值未改动。

## 检查 A

原验证服务日志有40条参考编码记录，去掉8条预热后，最后32条frames排序与离线JSONL前32行完全一致，`diff`为空。

[服务端frames](check_a/server_frames.txt) · [离线frames](check_a/offline_frames.txt) · [完整40条编码日志](check_a/all_server_encode_lines.txt) · [摘要](check_a/summary.json)

这是帧数多重集合一致，尚未证明codes内容逐元素一致。

## 检查 B

rows18/14，seed基数0/100/200/300/400，每个seed分别codes和audio，C1、0warmup、保存WAV。实际请求seed=基数+row。

新`analyze_audio.py`按默认10ms帧、相对17dB、long-pause≥0.7s运行：

| row | 模式 | ≥0.7s停顿的run | failed |
| ---: | --- | ---: | ---: |
| 18 | codes | 0/5 | 0 |
| 18 | audio | 0/5 | 0 |
| 14 | codes | 0/5 | 0 |
| 14 | audio | 0/5 | 0 |

loop-lag p99全部在0.18–0.94ms。最长检出的句中安静段为row18 codes实际seed118的0.41s。

[完整20行摘要](check_b/review_summary.txt) · [原始分析输出](check_b/audio_analysis.txt) · [分析JSON](check_b/audio_analysis.json)

## 为什么没有放行

| row / 模式 | 实际seed | 音频秒数 | RMS dBFS | Whisper small辅助识别 |
| --- | ---: | ---: | ---: | --- |
| 18 codes | 318 | 23.220 | -39.59 | Thank you. |
| 14 audio | 14 | 19.691 | -37.92 | Elvis Presley is often called. |
| 14 audio | 414 | 16.718 | -51.01 | Thank you. |
| 14 codes | 414 | 17.926 | -40.19 | Elvis Presley. |

同配置正常对照row18 codes、实际seed18完整识别目标句。两条“Thank you”可能是Whisper对低能量噪声的幻觉，不能当成实际可听内容或仅据此断言无语音；但结合能量、时长和正常对照，4条需要质量核查，且异常并非只出现在audio模式。

[下载4条异常+1条正常对照ZIP](check_b/outliers-and-control.zip)。全部20条原始WAV、summary、requests与chunk_log也保留在各run子目录。

- row18 codes seed318恰为1,024,000个采样，即500×2048，对应默认500帧生成上限。PCM响应和当前INFO日志没有finish_reason，未声称直接观测到`length`终止。
- 持续低能量片段使p95和相对quiet阈值一起降低，可能被全段判作voiced，因此“0 long-pause”不是语音正常的证明。
- 每个run只有1条样本，run内中位数等于自身，无法触发相对中位数的`slow`标记；words/s是目标词数除检测跨度，不是已识别出的实际语速。
- 这些输出的`waits_ms=[]`，异常在生成内容而非客户端等包；尚未定位模型内部具体根因，也未比较两份实际参考codes内容。

[能量证据](check_b/outlier_energy.json) · [500帧与分析器限制](check_b/length_and_analyzer_evidence.json) · [辅助ASR JSON](check_b/auxiliary_asr.json)

辅助ASR仅使用已有Whisper small、CPU、FP32，不是正式Seed-TTS WER或人工听检。没有修改原始summary或分析阈值来消除离群。

## 后续与清理

B1计划已明确为1088条codes、C1/16/32/48/64/96/128、C64×3/C128×5，共13次，每次重启、不保存音频。入口已准备但未执行；C128×256短测也未执行。需要先确认这批异常输出，再决定后续codes逐元素核对及放行。

本轮服务启动238.640秒，stage0加载30.479秒。API、stage、MPS、monitor已清理，8091释放，清理时GPU1回15MiB/0%；关停有已知semaphore清理警告。

[最终状态](assessment.json) · [运行参数](provenance.json) · [清理证据](cleanup.json) · [GPU记录](gpu.csv)

发布副本隐去本机绝对路径、hostname、GPU UUID；指标和音频字节保留。原始完整服务日志仍在本地。未改系统CUDA、驱动或服务器配置。
