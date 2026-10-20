# Drama 3 冒烟测试音频

本分支保存 2026-10-20 的 9 条模型生成音频，供人工试听。所有音频为单声道、16-bit PCM WAV。

**状态：功能检查通过，尚未听检。这次运行不是正式性能基线。**

## 测试来源

- 模型：`fishaudio/drama-3`，revision `b2b5ee998034b2975720906c815e7fb0e9f007bd`
- vLLM-Omni：[da572cddb38383deae9caef33b8d490e292ef071](https://github.com/advanced233/vllm-omni/commit/da572cddb38383deae9caef33b8d490e292ef071)
- GPU：NVIDIA A100 80GB PCIe
- 实际环境：vLLM 0.32.0+cu129、PyTorch 2.13.0+cu129、CUDA runtime 12.9、驱动 550.127.05；不满足官方 cu130 / R580+ 环境要求。
- 9 条正常请求均返回 HTTP 200；4 路并发全部成功；缺少 ref_text 的请求返回 HTTP 400。
- 采样率 44100 Hz；流式接口返回原始 PCM，客户端按 44100 Hz 协议假设封装 WAV。
- 原始请求指标见 [summary.json](summary.json)。TTFA、RTF 仅作本次记录。

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

克隆时长约 4–7 秒只是参考，短于 4 秒本身不代表失败。3.297 秒的 seed 0 样本需要听检确认内容完整；3.994 秒可视为约 4 秒。

文件校验值见 [SHA256SUMS](SHA256SUMS)。
