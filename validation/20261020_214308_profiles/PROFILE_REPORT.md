# Drama 3 A100：C8 / C16 / C64 实机 profiling

三档均完成全量1088条 codes 请求，不保存音频、不指定seed；全部请求成功。最终采用 C8 attempt2、C16 attempt1、C64 attempt1。C8 attempt1保留为首次采集诊断，其codec迭代预算将empty步计入，导致共同窗口过短，未用于最终比较。

实际graph probe证明：预先捕获的CUDA graph，三次replay各能记录并正确关联两个内部kernel。真实双stage/MPS trace也逐一核验：graph launch均有节点、无检测到的完全重复kernel或runtime correlation冲突，stage0未归因kernel累计时长均低于0.11%。日志未观察到CUPTI丢事件告警；这不是对任何潜在丢事件的绝对保证。

## 实际步骤与时长

步骤分类来自SchedulerOutput生成的原生context/generation CPU scope；剔除每线程首尾和不完整/无关联步骤。下列CPU scope是dispatch范围，GPU span是相关kernel首尾跨度，两者不能相加；也不能将不同热状态下的时间差都解释成并发效应。

| 客户端并发 | 完整decode / mixed | CPU scope p50，decode / mixed (ms) | GPU span p50，decode / mixed (ms) | trace SM median (MHz) | 温度范围 (°C) |
|---:|---:|---:|---:|---:|---:|
| 8 | 232 / 22 | 4.87 / 58.95 | 13.74 / 49.07 | 1410 | 82–83 |
| 16 | 194 / 29 | 6.28 / 58.11 | 18.06 / 46.32 | 1305 | 84–84 |
| 64 | 89 / 29 | 6.59 / 60.27 | 33.18 / 66.69 | 1155 | 84–85 |

实际decode batch分别为6–8、13–16、57–64；不能将C8/C16/C64标签当成所有步骤的固定batch。mixed的context请求数分别为1、1–3、1–7，prompt token预算也不同；完整直方图保存在selected_profiles.json。

## 可优先验证的优化方向

1. **Mixed步骤的GDN CPU执行路径。** 三档`vllm::qwen_gdn_attention_core`的CPU区间union p50约40.01 / 38.30 / 39.61ms，而整个mixed CPU scope约59–60ms。没有观察到同步API等待或D2H证据；54次graph launch本身也只占约数ms，不能把全部CPU区间归因于graph launch。应针对每层重复metadata准备、eager算子调度和graph边界做受控小实验，并用关闭profiler的端到端结果验证是否真实受益。Torch/CUPTI可能放大该CPU开销，因此这是优先验证的候选，不是已经证明的可消除时长。

2. **A100 Marlin MoE路径。** decode累计kernel时间占比随三档约30.94% / 38.60% / 46.12%；mixed约51.97% / 49.76% / 38.05%。原始Marlin名称及每层次数能与40层backbone的两次MoE GEMM对应，是归属最明确的GPU热点。可围绕实际batch/token分布做Marlin调度/分块参数或相关数据准备实验；任何收益需在同热状态、无profiler的基线和质量检查中确认。

3. **小batch dense GEMM/GEMV与BLAS reduction。** decode合计占比约38.30% / 33.13% / 23.32%。同名cuBLAS kernel不能区分backbone和codebook tower，当前不据此宣称tower占比；先补shape/调用归属，再选具体矩阵优化。

融合采样在当前decode中约2–3%，不是第一优先级。GDN的GPU占比低于其mixed CPU区间；不能因GPU占比小就排除CPU调度方向，也不能把提高GDN状态精度视作本次性能优化。

## 两阶段并发与限制

| 并发 | 共同有效采集窗口 (s) | stage0 / stage1 GPU busy重叠 (ms) | 重叠占共同窗口 |
|---:|---:|---:|---:|
| 8 | 4.423 | 393.1 | 8.89% |
| 16 | 5.036 | 626.4 | 12.44% |
| 64 | 5.109 | 1016.1 | 19.89% |

这些数值说明MPS下确有重叠，不能推导MPS让某阶段慢了多少，也不能仅据此判断codec是瓶颈。阶段各有独立profiler迭代预算，已按同host时钟与真实共同窗口对齐。

所有profile均保持原计算配置、私有MPS、相同CPU亲和性和用户cu129/R550环境；只增加profiler设置与CPU scopes。stage0最多256步、stage1最多4096步，客户端在请求阶段20秒后触发、5秒后请求停止。20秒业务稳态不等于热稳态；C16/C64采集时已出现软件热降频。profile吞吐与native导出暂停不进入B1。系统驱动/CUDA、公共软件和服务器源码未修改。

全部四次attempt各1088/1088成功，GPU1/8091及自有stage/MPS/监控进程均已释放；见final_cleanup.json。原始trace留在个人NVMe，逐步/原始kernel名称/CPU API证据见independent-audit，各执行参数见run_metadata.json与deploy_config.yaml。
