# GDN mixed 工具实机正确性门禁：未通过，停止第2/3步

2026-10-21，原字节工具提交 `daac4cf101e8a34010ac410e028aa1755ce3e6e2`，服务端 checkout `da572cddb38383deae9caef33b8d490e292ef071`，物理 GPU 1 / A100 80GB PCIe / cu129 / R550。

**第1步37项：26 passed、11 failed、0 skipped，退出码1。第2步微基准与第3步服务对照均未启动。** 未修改用户工具、夹具、断言、容差或服务源码；没有修补后重跑来覆盖失败。

## 先执行的源码指纹门禁

单独运行 `test_copy_is_the_installed_version`：1 passed，退出码0，10.50秒，之后才运行全套37项（19.77秒）。全套再次包含指纹测试。

当前与预期 normalized function SHA256 一致：

```text
9af0a98c23727bea4e7f34c1cf3283d838ad7a84a741b2fa8f946d823a4292a9
```

[指纹原始输出](step1/fingerprint.log) · [指纹JUnit](step1/fingerprint.junit.xml)

## 全套测试结果

| 检查组 | 通过 | 失败 |
| --- | ---: | ---: |
| 函数指纹 | 1 | 0 |
| baseline 与 installed 精确相等 | 8 | 2 |
| direct_out 与 installed 精确相等 | 10 | 0 |
| 与 vLLM 标准路径比较 | 1 | 9 |
| 非连续输出回退 | 2 | 0 |
| 两步续写 | 4 | 0 |
| 合计 | 26 | 11 |

- 两项 baseline 失败是 `prefill_only-fp32` 和 `prefill_only-bf16`，在测试第87行输出 `torch.equal(out, ref_out)` 处失败。打印的两侧输出均包含 NaN；不能通过允许 NaN 相等来把这两项改记为通过，日志也不足以归因于 direct_out。
- 标准路径10项中只有 `c64-bf16` 通过。另9项均先在 baseline 的输出 `assert_close`（第114行）失败，因此没有继续检查该用例的 direct_out 或后续池状态断言。`msg="baseline"` 覆盖了详细数值差异，原日志不能给出最大误差，也不能推断这9项都因 NaN 失败。
- 新夹具确实使用生产 `ChunkGatedDeltaRule.forward_native`，backend为Triton，并预计算chunk元数据。本轮不是此前旧夹具误走FlashInfer的问题，也没有模型权重加载或OOM失败。

**direct_out的10项installed对照通过，不等于整体正确性门禁通过，不能据此继续性能实验。**

[完整pytest输出](step1/full_tests.log) · [完整JUnit](step1/full_tests.junit.xml) · [结构化逐项结果](step1/summary.json) · [执行命令](step1/full_tests.sh)

## 静态诊断线索：slot 0 的约定

未进行修改或额外GPU重跑，但发现了值得先核对的夹具约定：

- 新 `harness.slots_for` 从0开始给真实请求分配slot；两步续写测试也显式使用 `[0, 1, 2]`。
- 当前安装的vLLM定义 `NULL_BLOCK_ID = 0`；packed recurrent decode对 `state_idx <= 0` 写零并返回，卷积kernel也跳过null block。
- checkout中的原上游fixture使用 `[1, 4, 7]` 等非零slot。

因此应优先检查真实请求是否错误占用了保留slot 0。调整时还需一并核对pool大小、两步续写的显式slot和“未使用行”断言。**这只是源码层面的线索，尚未验证它能解释全部11项失败，更不是生产模型故障的结论。**

[具体源码行号、片段及SHA](step1/static_slot_evidence.json)

## 环境、快照与清理

测试在指定checkout根目录、现有服务venv中执行。工具存为个人NVMe不可变实验快照，通过本地 `tools/drama3_bench` 符号链接接入；仅本地Git exclude排除该链接，服务端tracked源码保持干净。完整快照16文件SHA与提交一致，见 [源码锁定](source-lock.json) 和 [运行元数据](step1/metadata.json)。

GPU 1 已释放至15 MiB / 0%，无本轮计算进程。未启动服务，未修改驱动、CUDA、频率、功耗或风扇。

公开副本仅替换个人绝对路径、主机名和GPU UUID，测试错误与数值文字保持原样。原始证据保留在个人NVMe；[发布清单](publication.json)记录原始/公开哈希，根目录SHA256SUMS校验全部公开文件。
