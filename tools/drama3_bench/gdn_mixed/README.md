# GDN 混合步的 CPU 开销：低开销测量、微基准与正确性测试

profile 显示，在 A100 上含 prefill 的混合步里，GDN 核心（`fish_omni/gdn.py` 的 `mixed_forward_core`，每个混合步调用 30 次）的 CPU 作用域合计约 38–40 ms。但这个数是在 torch.profiler 下测的，包含 profiler 自身的开销。这里的工具做三件事：
1. 在不开 profiler 的情况下测出真实的 CPU 开销；
2. 把开销拆到各个阶段；
3. 测试第一个候选改动（让 FLA 直接写输出缓冲）：数值是否完全不变、能省多少。

| 文件 | 作用 |
| --- | --- |
| `drama3_gdn_variants.py` | `mixed_forward_core` 的拷贝（来自 da572cd），逐行保持原有工作，额外支持两样东西：每个阶段的 `perf_counter_ns` 计时（只记 CPU 时间，不同步 GPU）；`direct_out` 变体（非 FlashInfer 路径上，把 `core_attn_out[decode:tokens]` 作为扁平视图交给 FLA，省掉一次输出分配和复制）。附带源码指纹检查，确保拷贝的版本和已安装的一致 |
| `drama3_gdn_harness.py` | 构造一个 GDN 层、状态池和混合步元数据。走的是 A100 实际跑的 FLA/Triton 路径，并像生产代码一样预先算好 chunk 元数据 |
| `test_mixed_core_variants.py` | 正确性测试（pytest） |
| `bench_mixed_core.py` | 单层微基准，不需要起服务 |
| `drama3_gdn_probe.py`、`probe_site/sitecustomize.py` | 服务内探针：不改源码，进程启动时把 `mixed_forward_core` 换成带计时的拷贝，按混合步记录 |
| `summarize_probe.py` | 汇总探针记录 |

以下命令都在 **vllm-omni 仓库 checkout（da572cd）的根目录**、激活服务用的 venv 后执行。`GPU`、`REPO` 换成你的值。

## 第 1 步：正确性测试（先跑，必须全部通过）

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest -v tools/drama3_bench/gdn_mixed/test_mixed_core_variants.py
```

一共 37 项：
- **`test_copy_is_the_installed_version`**（1 项）：已安装的 `mixed_forward_core` 和拷贝的指纹必须一致。不一致就说明装的不是 da572cd，后面的步骤都不要做。
- **`test_variant_equals_installed_bit_for_bit`**（20 项）：baseline 和 direct_out 两个变体，在 5 种步形和 BF16/FP32 两种状态精度下，输出、卷积池、SSM 池都要和已安装函数**逐位一致**。同时检查输出缓冲里步之外的行、状态池里没用到的行都没被改动。
- **`test_variants_match_vllm_standard_path`**（10 项）：用上游测试的容差，和 vLLM 的标准 `_forward_core` 对比。
- **`test_non_contiguous_output_takes_the_copy`**（2 项）：输出缓冲不连续时，direct_out 必须退回复制，结果仍然一致。
- **`test_prompt_split_over_two_steps`**（4 项）：同一个 prompt 分两步 prefill（先 fresh、再在同一行 continued），两步结束后结果一致。

步形取自 profile 实测：C8 是 7 个 decode + 170 个 prefill token；C16 是 15 + 176；C64 是 60 + (180 + 178)；另有上游测试原样的步形、只有 prefill 的步，以及 8 + 512 个 continued token。fresh prefill 落在留有旧状态的行上，同时覆盖了"行被复用"的情况。请求从状态池第 1 行开始分配：第 0 行是 vLLM 的 null block（`NULL_BLOCK_ID`），调度器从不把它分给请求，kernel 遇到它会跳过或直接写零；夹具若把它分给请求会直接报错。

这里不涉及 CUDA graph 回放：混合步的 GDN 核心是 eager 执行的 custom op，本来就不在 graph 里。

## 第 2 步：微基准（单层，不用起服务）

```bash
CUDA_VISIBLE_DEVICES=1 taskset -c 0-27 python tools/drama3_bench/gdn_mixed/bench_mixed_core.py \
    --cases c8 c16 c64 --state-dtype bf16 --iters 200 --kernel-time --json results/gdn_bench_bf16.json
```

- `taskset` 用和服务 stage 0 相同的 CPU 核，因为测的就是 CPU 时间。
- 先确认第一行输出是 "matches"。
- 每一轮会恢复状态池，然后三个变体按随机顺序各调用一次，让它们处在相同的热状态下。

结果表怎么读：
- **host us**：一次调用在 CPU 上发射全部工作所花的时间。调用前 GPU 是空闲的，调用中没有同步。
- **GPU span us**：用 CUDA events 包住一次调用的时长，里面包括 GPU 等 CPU 发 kernel 的空闲时间。span 和 host us 接近，说明这次调用受 CPU 限制。
- **kernel us**（`--kernel-time`）：在单独一轮里用 torch.profiler 统计的 kernel 设备时间之和。profiler 会抬高 CPU 时间，但不怎么影响 kernel 时长。
- **host ms ×30**：乘以 30 个 GDN 层，就是一个混合步的估算值，可以直接和 profile 的 38–40 ms 比较。
- **installed 和 baseline 应该基本相同**。如果差很多，说明拷贝本身带来了额外开销，探针的数字就要打折扣。
- **direct_out 比 baseline 少的部分**，就是候选改动能省下的时间。下方还有各阶段的中位数。

## 第 3 步：在服务里测（不开 torch.profiler）

服务进程的 `PYTHONPATH` 里加上 `probe_site`，同时设 `DRAMA3_GDN_PROBE=1`。stage 子进程会继承这些环境变量。

```bash
P=$REPO/tools/drama3_bench/gdn_mixed/probe_site
PYTHONPATH=$P${PYTHONPATH:+:$PYTHONPATH} DRAMA3_GDN_PROBE=1 DRAMA3_GDN_VARIANT=baseline \
DRAMA3_GDN_PROBE_DIR=results/probe_c16_baseline \
    vllm serve <模型> --omni --port 8091 ...      # 和 B1 相同的启动命令与 CPU 亲和性
```

- **确认生效**：服务日志里必须有 stage 0 进程打出的 `[drama3-gdn-probe pid=...] patched mixed_forward_core`。API server 进程可能也会打一条，没有影响。看到 `NOT patched` 就停下来检查。
- **压测**：用 `stream_bench.py` 跑 C16。为了避开热降频，可以跑短一点（`--num-prompts 256`），或者用分段模式，输出放到 `--out results/c16_probe_baseline`。跑完停掉服务，探针在退出时会写出最后一批记录。
- **汇总**：

  ```bash
  python tools/drama3_bench/gdn_mixed/summarize_probe.py results/probe_c16_baseline \
      --bench-summary results/c16_probe_baseline/summary.json
  ```

  `--bench-summary` 只保留压测计时窗口内的步，启动时的 CUDA graph 捕获和 warmup 都会排除掉。

建议跑三次对照：
1. **`DRAMA3_GDN_PROBE_LAPS=0`**：只记整次调用的时长，不分阶段。和分阶段计时那次比较，可以确认分阶段计时本身不增加开销。按设计，每步的额外开销只有几十微秒。
2. **不设 `DRAMA3_GDN_PROBE`**：比较端到端的 TTFA/RTF，确认探针对整体没有影响。
3. **`DRAMA3_GDN_VARIANT=direct_out`**：第 1 步全部通过之后再跑，测候选改动在真实服务里的效果。

汇总表怎么读：
- **GDN host ms/step**：一个混合步里 30 次调用的 CPU 时间之和。这就是去掉 profiler 后的真实值，和 profile 的 38–40 ms 比较。
- **span ms**：从第一次 GDN 调用开始到最后一次结束的时长，大致覆盖整个 backbone 部分在 CPU 上的时间。GDN 之和占 span 的比例越大，越说明 GDN 是混合步 CPU 时间的主体。
- **分阶段表**：
  - `chunk` 是 FLA 那一串 Python 加 Triton 发射；
  - `gather`、`scatter` 是状态池的取出和写回；
  - `copy` 是 direct_out 要省掉的那次复制；
  - `conv`、`decode`、`prep` 是其余几个 kernel 的发射。

  它决定下一步该改哪里。

`run_sweep.sh` 里可以把这些环境变量直接写进 `SERVER_CMD`。

## 交叉验证：py-spy

```bash
py-spy record -p <stage0 pid> -r 200 -d 20 --native -o stage0_c16.svg
```

stage 0 的 pid 在服务日志 `StageEngineCoreProc_stage0_replica0 pid=...` 里。py-spy 完全不插桩，只做采样，可以用来独立核对探针的结论。它需要 ptrace 权限，可能要 root，或者 `kernel.yama.ptrace_scope=0`。

## 局限
- 计时用的是 `perf_counter_ns`，不做同步。如果 GPU 队列满了，发射本身会阻塞，这部分等待也会算进 CPU 时间。混合步受 CPU 限制，一般不会出现这种情况，可以用微基准里 GPU span 和 host us 的关系来核对。
- 探针只能看到混合步。纯 decode 步是整步 CUDA graph 回放，不经过 Python。
- 进程被 SIGKILL 杀掉时，最多丢失最后 31 步的记录。
- **校验范围**：探针的导入钩子、指纹不一致时拒绝打补丁、按步聚合、批量写出、对已有 `sitecustomize` 的串联，以及汇总脚本，都用替身模块测过。用到 GPU 的部分（拷贝函数、测试夹具、微基准、正确性测试）**还没在 GPU 上运行过**，请按顺序从第 1 步开始。
