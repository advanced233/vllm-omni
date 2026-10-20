# Drama 3 压测工具（按 #8933 的测法）

用 PR [vllm-project/vllm-omni#8933](https://github.com/vllm-project/vllm-omni/pull/8933) 的测法在 A100 上测 Drama 3 的流式吞吐、首包延迟、RTF 和卡顿率。#8933 在基线和优化后两边都把参考音色预编码成 `ref_audio_codes` 发送（"so neither side encodes references"），但没有公开预编码脚本和压测客户端。这里补上这两样，外加扫描和汇总脚本。

| 文件 | 作用 | 运行环境 |
| --- | --- | --- |
| `encode_ref_codes.py` | 把 Seed-TTS-Eval 每行的参考音频离线编码成 `ref_audio_codes` | vLLM-Omni 环境 + CUDA GPU |
| `stream_bench.py` | 闭环流式压测一个并发档位，输出 `summary.json` / `requests.jsonl` | 只需要 `aiohttp`、`numpy`（vLLM 环境里都有；有 `uvloop` 会自动使用） |
| `run_sweep.sh` | 多档并发扫描，每次都重启服务，同时记录 GPU 时钟、功耗和降频原因 | bash、curl、nvidia-smi |
| `aggregate_results.py` | 把多次运行汇总成 #8933 那样的表格，重复运行取平均 | Python 3.10+ |
| `analyze_audio.py` | 排查音频：卡死（持续底噪）、语速、句中停顿，以及停顿是否落在 codec 窗口边界上 | `numpy` |
| `eval_quality.py` | 对 `--save-audio` 保存的回复算 Seed-TTS WER 和说话人相似度（SIM），并按是否卡死拆分 | vLLM-Omni 环境（`pip install 'vllm-omni[dev]'` 带的 torch、transformers、jiwer、zhon、scipy、soundfile） |

## 1. 预编码参考音频

```bash
python tools/drama3_bench/encode_ref_codes.py \
    --model ${NVME_WORKDIR}/models/fishaudio/drama-3 \
    --seed-tts-root /path/to/seed-tts-eval --locale en \
    --out ${WORKDIR}/ref_codes_en.jsonl
```

- 解码和编码都调用服务端处理 `ref_audio` 时用的同一套代码：vLLM 的 `MediaConnector` 解码 wav 并取单声道（`serving_speech._resolve_ref_audio_array`），再用 `fish_speech.dac_encoder.encode_reference_audio_codes` 以 fp32 编码（stage 0 处理 `ref_audio` 请求时调用的就是它，见 `fish_omni_model.py` 的 `_reference_codes`）。结果从 `[frames, codebooks]` 转置成 API 要求的 `[codebooks, frames]`。
- 默认在 `cuda` 上编码，和 stage 0 一致；CPU 编码在极少数最近邻并列的情况下可能选出不同的 code。编码器要占几 GiB 显存，所以先编码，再起服务。
- 和服务端一样，参考音频时长必须在 1–30 s 之间；每个值也按 fish_omni adapter 的规则检查（第 0 个 codebook 取值 < 4096，其余 < 1024）。不合格的行会跳过，跳过原因写进 meta。
- 输出：
  - `ref_codes_en.jsonl`：每行一条，含 `utt_id`、`ref_text`、`target_text`、`prompt_wav`、`frames`、`ref_audio_codes`。
  - `ref_codes_en.jsonl.meta.json`：记录 `codec.pth` 和 `meta.lst` 的 sha256、设备名、vLLM-Omni commit、各包版本、跳过的行。压测时这些信息会一并写进 `summary.json`。
- Seed-TTS-Eval en 应该写出 1088 行。条数不对先查 meta 里的 `rows_skipped`。

## 2. 单档压测

先起服务（和冒烟时同一条命令，默认 deploy 是 `fish_omni.yaml`），然后：

```bash
python tools/drama3_bench/stream_bench.py \
    --codes ${WORKDIR}/ref_codes_en.jsonl --concurrency 64 \
    --base-url http://127.0.0.1:8091 --model fishaudio/drama-3 \
    --label a100-main --meta commit=da572cdd --out results/c64
```

结束时会打印一行和 #8933 recipe 同格式的表格。有请求失败时退出码为 2。

## 3. 扫描（每次重启服务）

```bash
SERVER_CMD='source ${PERSONAL_WORKDIR}/activate_cuda129.sh && CUDA_VISIBLE_DEVICES=1 vllm serve ${NVME_WORKDIR}/models/fishaudio/drama-3 --omni --port 8091 --served-model-name fishaudio/drama-3' \
CODES=${WORKDIR}/ref_codes_en.jsonl GPU_INDEX=1 LABEL=a100-main \
CONCURRENCY="1 16 32 48 64 96 128" REPEATS_AT="64:3 128:5" \
CLIENT_PREFIX="taskset -c 60-63" \
bash tools/drama3_bench/run_sweep.sh
```

- `SERVER_CMD` 在前台启动服务，脚本把它放进独立进程组。每次运行依次是：等 `/health` 就绪 → 压测 → 停服务（先 SIGINT，不行再 SIGTERM，最后 SIGKILL）→ 等显存降到 `GPU_IDLE_MIB` 以下、端口释放，再开始下一次。
- `REPEATS_AT="64:3 128:5"` 对应 #8933 的"C64 / C128 are the mean of 3 / 5 runs"。
- `GPU_INDEX` 填 nvidia-smi 看到的物理编号或 UUID，要和 `CUDA_VISIBLE_DEVICES` 指向同一张卡。每次运行的 `gpu.csv` 记录 SM/显存时钟、功耗和功耗上限、温度、利用率、显存，以及降频原因（较新驱动叫 `clocks_event_reasons`，旧驱动叫 `clocks_throttle_reasons`）。
- `CLIENT_PREFIX` 用来把客户端绑到和服务端不同的 CPU 核上，避免互相抢占。
- 换 deploy 变体时，在 `SERVER_CMD` 里加 `--deploy-config <yaml>`，同时换一个 `LABEL`。
- 输出：`$OUT/c<并发>/r<第几次>/` 下有 `server.log`、`client.log`、`gpu.csv`、`summary.json`、`requests.jsonl`、`startup_seconds.txt`、`cleanup.json`；`$OUT/table.md` 是汇总表。

## 4. 汇总

```bash
python tools/drama3_bench/aggregate_results.py results/sweep_a/ results/sweep_b/ --json table.json
```

每个（label, 并发）一行，重复运行取平均，括号里是 min–max。

## 5. 排查个别样本（语速慢、停顿）

```bash
# 只重发第 18 行（即 WAV 文件名前缀 0018），并发 1，换几个 seed，两种参考方式都跑
for seed in 0 100 200 300 400; do
  python tools/drama3_bench/stream_bench.py --codes ${WORKDIR}/ref_codes_en.jsonl --concurrency 1 \
      --rows 18 --seed $seed --num-warmups 0 --save-audio --out results/row18_codes_s$seed
  python tools/drama3_bench/stream_bench.py --codes ${WORKDIR}/ref_codes_en.jsonl --concurrency 1 \
      --rows 18 --seed $seed --num-warmups 0 --save-audio --ref-mode audio \
      --seed-tts-root <seed-tts-eval 路径> --out results/row18_audio_s$seed
done
python tools/drama3_bench/analyze_audio.py results/check_codes results/check_audio results/row18_* \
    --codes ${WORKDIR}/ref_codes_en.jsonl --seed-tts-root <seed-tts-eval 路径> --all
```

`analyze_audio.py` 对每条回复报告：
- 卡死（stuck take）：模型不再说话，持续输出平稳底噪，常常一直到 500 帧上限（23.22 s）。这种底噪可能和轻声说话一样响（-38 到 -50 dBFS），任何能量阈值都分不开；它的特点是平：每个 1 s 窗口内 10 ms 帧电平的 p90−p10 不到 10 dB，而语音每个音节都有 10–30 dB 的起伏。连续平坦 ≥ 2 s 标记 `noise-run`，达到帧数上限标记 `max-frames`。每个 run 和合计都会打印 noise-run 的比例。在最早 84 条 A100 回复上，卡死的几条连续底噪 16.5–22.8 s，其余最多 1.25 s。
- 语速：目标文本词数除以检测到的语音跨度（不是转写出的实际词数）；如果给了参考音频，也报参考音频的语速。`slow` 和所有回复（去掉 noise-run）合起来的中位数比较，所以单条一个 run 也能触发。
- 句中停顿：以 10 ms 为一帧，比响亮帧（第 95 百分位）低 17 dB 以上的帧算安静帧；连续安静 ≥ 0.25 s 算一次停顿，≥ 0.7 s 标记为 long-pause。Drama 3 的停顿不是数字静音，而是 -45 dBFS 左右的底噪（语音本身在 -20 到 -25 dBFS），所以用相对阈值。
- 停顿两端到最近 codec 窗口边界的距离。`requests.jsonl` 里有 `chunk_log`（本版本起才记录）时，用实际收到的字节突发位置作为边界；没有时按 ramp 推算（6, 8, 12, 16, 24, 32, 之后每 64 帧，每帧 2048 个采样）。

判读方法：
- 语速慢不可能是 codec 造成的，每帧固定解码成 2048 个采样。语速只取决于 stage 0 每个词采样了多少帧，所以语速问题要到参考音频、采样或数值精度上找。
- 停顿的边缘落在窗口边界附近（默认 25 ms 以内）时，会标记 on-boundary。单个样本这样可能只是巧合，因为窗口边界之间相隔 0.3–3 s；但如果多个样本都这样，问题就在 codec 的流式拼接上。

## 6. 质量评测（WER / SIM）

```bash
# 质量运行：保存音频（不要和纯性能运行混用）
python tools/drama3_bench/stream_bench.py --codes ${WORKDIR}/ref_codes_en.jsonl --concurrency 16 \
    --seed 0 --save-audio --label quality-default --out results/quality_default
# 服务停掉后（或换一张卡）再评测
python tools/drama3_bench/eval_quality.py results/quality_default \
    --seed-tts-root <seed-tts-eval 路径> --device cuda:0
```

- 调用的是仓库评测器 `vllm_omni/benchmarks/data_modules/seed_tts_eval.py` 里的同一套函数：重采样到 16 kHz、`openai/whisper-large-v3`（fp32）转写、seed-tts-eval 的文本归一化和 WER 计算，口径和 `vllm bench serve --seed-tts-wer-eval` 一致。一次运行的 WER 是逐句 WER 的平均（同 seed-tts-eval 的 `run_wer.py`）。
- 转写为空的句子按 WER 1 计（相当于每个词都被删掉）。仓库评测器会直接跳过这些句子，所以另外报一个 `wer_mean_nonempty`，用来和 `vllm bench serve` 的数字对照。
- SIM 用的是仓库的代理指标：WavLM（默认 `microsoft/wavlm-base-plus`）均值池化后的余弦相似度，不是 seed-tts-eval 官方的 WavLM-large 说话人验证模型。只能在本工具的不同运行之间比较，不能和论文或模型卡里的 SIM 数字比。
- 每句都会用 `analyze_audio` 的卡死检测打标记，所以同时报"去掉卡死句后的 WER / SIM"和卡死比例；`at cap` 是达到 500 帧上限的句数。
- 输出写到各 run 目录下的 `quality.json`，包含汇总和每一句的转写、WER、SIM、卡死标记。终端会打印汇总表和 WER 最高的几句。
- 1088 句在 A100 上估计要 10–20 分钟（Whisper-large-v3 fp32，未实测），模型加载会占几 GiB 显存。评测时服务要么已经停掉，要么在另一张卡上。

## 口径（和 #8933 对照）

| 项 | #8933 原文 | 本工具 |
| --- | --- | --- |
| 数据 | Seed-TTS-Eval en，1088 行，每行用自己的参考音频，以 `ref_audio_codes` 发送 | 相同（`--ref-mode codes`，默认） |
| 请求 | streaming `/v1/audio/speech`（raw pcm），默认采样 | `response_format=pcm, stream=true, stream_format=audio`；不传采样参数，用服务端默认值（temperature 1.0 / top_p 0.9 / top_k 30 / max_tokens 500） |
| 并发 | 闭环，每档重启服务 | `--concurrency` 个请求始终在途，直到数据跑完；`run_sweep.sh` 每次运行都重启服务 |
| warmup | 排除 8 条 warmup | 正式计时前先跑 8 条（取数据前 8 行，seed 另加偏移），不计入结果 |
| RTF | 从发请求到最后一个音频块的时间 ÷ 音频时长 | 相同，时间戳在客户端收到字节时记录 |
| TTFA | — | 从发请求到收到第一个完整采样 |
| 卡顿 | "A stream stalls when its next audio arrives after the audio already received has played out" | 从第一段音频开始播放；下一段音频到达时，已收到的音频已经播完，就算一次等待。同时报 >0 ms（#8933 口径）和 >100 ms（`vllm bench serve` 默认口径）两档 |
| 吞吐 | 生成的音频秒数 ÷ 墙钟秒数 | 相同（从第一个请求发出到最后一个响应结束）；另外报"满并发窗口"吞吐（最后一行派发之前，所有槽位都忙的那段时间），去掉了收尾时并发逐渐降低的那部分 |
| 重复 | C64 / C128 分别取 3 / 5 次平均 | `REPEATS_AT` + `aggregate_results.py` |

## 注意

- **客户端时间戳的误差**：`summary.json` 里的 `client.loop_lag_ms` 是 asyncio 事件循环被唤醒的延迟，所有时间戳都带着这个误差。p99 超过 10 ms 时脚本会告警，这时卡顿判定不可信，应该把客户端换到空闲的 CPU 核上。计时开始前会冻结 GC，避免 GC 停顿混进时间戳。
- **`--ref-mode audio`** 把参考 wav 以 base64 内联成 `ref_audio` 发送（需要 `--seed-tts-root`），服务端会在 stage 0 现场编码。拿它和 codes 模式对比，可以量化"每个请求现编码"的开销。它不是 #8933 的测法。
- **seed 不能复现输出**：`--seed` 会给每个请求设 `seed = 基数 + 行号`，但 bf16 logits 随 batch 组成而变（#8933 评审里提到过），并发下同一个 seed 也不保证输出相同。评估质量时应该用多次运行看方差。
- **WER / SIM**：用第 6 节的 `eval_quality.py`。保存音频会占用客户端的 IO，只建议在测质量的那次运行里打开，不要在纯测性能的运行里用。
- **校验范围**：这些脚本在本地用一个模拟服务器验证过，覆盖了吞吐、窗口内的音频计算、卡顿判定（人为制造的卡顿）、500 错误和流中途断开、warmup 排除、扫描时的进程启停和信号升级。`encode_ref_codes.py` 的数据解析、去重、跳过逻辑和输出格式用假编码器测过。**两者都还没在真实模型和 GPU 上跑过**。第一次正式运行前，先用 `--num-prompts 32` 小规模跑一次，核对 `summary.json`。

## 实机验证记录

2026-10-20，工具执行版本 `49599d4` 已在A100/cu129/R550上完成预编码1088条和codes/audio各C4 × 32验证。两组failed=0，loop-lag p99=0.85/0.87ms，664项数据一致性检查通过。音色和时长长尾仍待人工听检，尚未进入正式基线。上面的“只在模拟服务器验证”描述的是本次实机运行前的状态。

[实机结果与试听样本](../../validation/20261020_165001_49599d4/README.md)
