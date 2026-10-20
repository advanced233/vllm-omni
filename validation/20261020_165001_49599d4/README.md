# Drama 3 工具实机验证（C4 × 32）

**数值与数据一致性检查通过；时长差异和音色听检待完成。没有开始正式基线。**

工具执行版本 `49599d4a438062da8c7799243a1291fc52c9b3d0`；服务源码 `da572cddb38383deae9caef33b8d490e292ef071`。
A100 80GB PCIe GPU 1，PyTorch 2.13.0+cu129 / vLLM 0.32.0+cu129 / 驱动550.127.05；不代表官方cu130/R580环境。

## 验收结果

- 预编码 **1088/1088**，`rows_skipped=[]`、退出码0，见 [预编码meta](ref_codes_en.jsonl.meta.json)。
- codes / audio 两组各 **32/32成功、failed=0**；相同前32行、每条seed=0+行号、各8条不计时预热、并发4、保存音频。
- loop-lag p99：**0.85 / 0.87 ms**，均低于10ms。
- 独立核验 **664项通过**：WAV帧数/原始字节数/44100Hz单声道16bit/时长/RTF/汇总统计相符。不能从已保存records重建所有原始到达时序，因此没有声称独立验证每个网络时间戳或满并发窗口吞吐。
- 两组音频时长中心接近，但长尾不同，不能把“failed=0”视为音色或语音质量通过。

| 模式 | 失败数 | loop-lag p99 ms | 音频mean / p50 / p95 s | TTFA p50 / p95 ms | mean RTF |
| --- | ---: | ---: | --- | --- | ---: |
| codes | 0 | 0.85 | 3.844 / 3.599 / 5.821 | 134.3 / 184.4 | 0.2837 |
| audio | 0 | 0.87 | 4.229 / 3.460 / 8.306 | 211.0 / 369.6 | 0.3359 |

[check_codes/summary.json](check_codes/summary.json) · [check_audio/summary.json](check_audio/summary.json) · [汇总](validation-summary.json) · [独立核验](independent_validation.json)

保存音频会影响运行开销；这些是工具验证数据，不是正式性能基线。相同seed在不同batch组成下也不保证生成完全相同。

## 四组待听检样本

先听参考，再比较codes和audio输出是否音色一致、内容完整、没有杂音或异常停顿。选择包含差异最大的18、14、2及常规样本0。全部64条生成WAV也保留在两组audio目录，可自由扩大抽查。

| index | 参考音频 | codes输出 | audio输出 | 目标文本 |
| ---: | --- | --- | --- | --- |
| 0 | [参考](listening/reference_0000.wav) | [codes 2.183s](check_codes/audio/0000_common_voice_en_10119832-common_voice_en_10119840.wav) | [audio 2.276s](check_audio/audio/0000_common_voice_en_10119832-common_voice_en_10119840.wav) | Get the trust fund to the bank early. |
| 2 | [参考](listening/reference_0002.wav) | [codes 5.898s](check_codes/audio/0002_common_voice_en_103675-common_voice_en_103676.wav) | [audio 8.034s](check_audio/audio/0002_common_voice_en_103675-common_voice_en_103676.wav) | One by one, the campfires were extinguished, and the oasis fell as quiet as the desert. |
| 14 | [参考](listening/reference_0014.wav) | [codes 2.786s](check_codes/audio/0014_common_voice_en_137148-common_voice_en_137152.wav) | [audio 5.480s](check_audio/audio/0014_common_voice_en_137148-common_voice_en_137152.wav) | Elvis Presley is often called the "King of Rock and Roll". |
| 18 | [参考](listening/reference_0018.wav) | [codes 4.133s](check_codes/audio/0018_common_voice_en_15265-common_voice_en_15266.wav) | [audio 9.799s](check_audio/audio/0018_common_voice_en_15265-common_voice_en_15266.wav) | Large brown dog walks up a blue staircase. |

Whisper small CPU对这8条输出的辅助转写均与目标文本匹配（简单规范化后的词编辑距离0）。但audio index18在3.28–7.58s有约4.30秒连续低能量区间（20ms窗RMS<-40dBFS）；index14也有较长句内低能量区间。它们不能仅凭RMS认定为静音、杂音或某种发音，也不能用ASR确认音色。

这是辅助诊断，不是正式Seed-TTS WER或人工听检。详见 [selected_asr_checks.json](selected_asr_checks.json)、[时长差异排序](duration-outliers.json)、[最终试听manifest](listening/manifest.json)。独立核验器的原始报告含最初固定抽样0/10/20/31，发布试听包使用本表，以覆盖差异样本。

## KV容量

[原始grep日志行](kv_cache_lines.txt)：

```text
GPU KV cache size: 326,716 tokens, Maximum concurrency for 8,192 tokens per request: 39.88x
```

默认`max_num_seqs=128`。按服务端相同prompt构造计算，1088条prompt为121–264行；加生成上限500为621–764行。39.88x按8192 tokens/request计算，不能据此直接判定短请求C128放不下，也不能按简单比例保证C128。

混合Mamba/KV组、页对齐及运行时workspace仍需考虑。本轮仅实测C4，没有执行C128或修改部署yaml；需听检确认后再决定C128短压测和正式基线。详见 [容量分析](prompt_capacity.json)。

## 数据来源与参考音频署名

数据来自 [zhaochenyang20/seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval/tree/8f5e1aa2a35d42f42e940074c1983358b9491f89) 的固定版本`8f5e1aa2a35d42f42e940074c1983358b9491f89`；这是 [ByteDance Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval) 的第三方整理副本，英文源为Common Voice。数据卡标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本目录4条参考WAV按原样复制，仅重命名；原路径与哈希见试听manifest，权利与署名归原数据作者。

下载的metadata含1088条、666个不同参考WAV，668个源文件全部通过HF LFS SHA256或Git blob SHA1校验。未与官方Drive原始tar做字节级对照。详见 [数据来源及校验](dataset-provenance.json)。预编码JSONL本体未上传，元数据和其SHA256可用于复现核对。

## 运行与清理记录

[完整文字报告](report.txt) · [运行元数据](run-metadata.json) · [部署配置](deploy_config.yaml) · [启动日志摘要](key_lines.txt) · [GPU记录](gpu.csv) · [清理记录](cleanup.json)

服务启动357.952秒；运行阶段无ERROR/Traceback。关停有内部force-kill和semaphore清理警告，最终服务、MPS、监控及端口均释放，GPU1回15MiB/0%。

本机绝对路径、hostname和GPU UUID已在发布副本中替换为占位符，数值、音频字节及原始指标保持不变。完整server日志保留本地，本目录公开摘要；文本报告中提到的其他本地辅助文件未全部上传。
