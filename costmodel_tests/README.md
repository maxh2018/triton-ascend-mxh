# CostModel 测试用例与脚本使用说明

本分支 `costmodel/test-cases` 用于保存和持续扩充 CostModel 测试资料。
目标仓库为 <https://github.com/maxh2018/triton-ascend-mxh>。

## 收录范围和来源

| 测试集 | 路径 | 本次收录 |
|---|---|---|
| pytest_ut 全集 | `third_party/ascend/unittest/pytest_ut/` | 保留目标 main 原有 361 个 `test_*.py`，补入 PR 快照独有的 13 个文件，共 374 个测试文件 |
| PR #2528 回归矩阵 | `third_party/ascend/unittest/costmodel_ut/costmodel_generalization/` | 100 个具名 case、19 个 benchmark 驱动，以及 FZE/solve_tril kernel |
| PR 测试脚本 | 同上 | 运行、审计、分片、合并、合同校验、残差诊断、profile 采集和后端证据采集脚本 |
| Profile 快照 | `third_party/ascend/costmodel/profiles/` | SIMD/SIMT selection profile、microbench JSON 和 schema；包含 pytest 校准测试所需 validator |
| 新增统一入口 | `costmodel_tests/run.py` | pytest 全集、矩阵全量、单 case、组合执行及命令预览 |
| 新增单 case 入口 | `costmodel_generalization/run_generalization_case.py` | 自动生成单 case 清单、运行全部规定路线并审计 |

基线 main：`a176ab28b11b63d427760c15f6cd63c14ac296f1`。
导入 PR：<https://github.com/triton-lang/triton-ascend/pull/2528>，固定提交
`4a74fd38ec69836a572b2f8d664f70c7e2fe3dfd`。
详细文件来源见 [sources.json](sources.json)。保留 main 现有同名 pytest 文件，不用旧 PR 版本覆盖。
374 是文件数，pytest 参数化后的测试数量需要在真实环境执行 collection 才能得到。
pytest 集和矩阵存在算子语义重叠，不能把两者数量相加当作独立算子数。

矩阵里的 100 个 case 来自历史 71 + 17 + 12；同一 kernel 的多个 shape/program 配置可占多个 case。
benchmark 的 `source_pytest` 字段表示改编来源，实际执行的通常是 benchmark 中的 kernel 副本。
本分支保存测试和 profile，不合并 PR 的 C++ CostModel、backend/compiler 实现或 NPU-IR 改动。

## 环境要求

在具有空闲 Ascend NPU 的 Linux 主机上使用，先加载自己的 CANN、驱动、Python、
Triton-Ascend 和 bishengir 环境。Python 建议使用原 PR 的 3.11 环境。
普通 pytest 测试需要 `pytest`、`pytest-xdist`、`torch`、`torch_npu`、Triton；
部分测试还有额外依赖（例如 MindSpore），应按被测文件安装，不能把 collection 失败当作通过。
原 PR 的离线残差诊断另需 `numpy`。

**矩阵需要安装支持 PR #2528 compile options 和 route JSON 格式的 Triton-Ascend，以及匹配的
bishengir/NPU-IR。直接安装本分支继承的 main 编译器，不保证能运行 PR 测试。**
先准备编译器环境，再把本分支作为测试代码使用。默认 profile 对应 950PR/A5；950DT 快照
仅为 bootstrap，不可用它声称完成生产 route selection 验收。其他目标需显式指定兼容 profile。

```bash
git clone --branch costmodel/test-cases https://github.com/maxh2018/triton-ascend-mxh.git
cd triton-ascend-mxh

# 按你的服务器实际路径加载环境，例如：source /path/to/your/set_env.sh
python -c 'import torch, torch_npu, triton; print(triton.__file__)'
command -v bishengir-compile
npu-smi info
```

不要设置覆盖所有路线的 `TRITON_ASCEND_COMPILE_MODE`。默认不自动安装、构建或升级编译器。
下列命令从仓库根目录执行，`--device 0` 表示使用物理卡 0，可改为空闲卡号。

## 先查看清单与执行计划（不需要 NPU）

```bash
python costmodel_tests/run.py list
python costmodel_tests/run.py case --case softmax_original --device 0 --dry-run
python costmodel_tests/run.py matrix --device 0 --dry-run
```

## pytest_ut：运行所有原始测试

```bash
python costmodel_tests/run.py pytest --device 0

# 单个测试/筛选：--pytest-args 必须放在最后
python costmodel_tests/run.py pytest --device 0 --pytest-args -k atomic
python costmodel_tests/run.py pytest --device 0 --pytest-args --collect-only
```

返回码和 JUnit XML 表示 pytest 计算精度、编译或接口断言的结果。
这批原始测试没有统一的 CostModel 耗时预测接口，**不自动进行 ±10% 预测误差验收**。
collection 时也可能需要导入 NPU 运行库。并行会争用设备和影响计时，性能矩阵默认串行。

## PR 矩阵：运行并验收单个 case

```bash
python costmodel_tests/run.py case --case softmax_original --device 0
python costmodel_tests/run.py case --case atomic_discrete_i32_p1_b256 --device 0
```

也可以直接使用单 case 脚本（支持 `--warmup`、`--repeats`、`--measurement-processes`）：

```bash
python third_party/ascend/unittest/costmodel_ut/costmodel_generalization/run_generalization_case.py \
  softmax_original --device 0
```

case 参数来自矩阵的 `name`，不能填 `bench_*.py` 文件名。
脚本自动运行 Auto、SIMD、SIMT F1/F2/F4；8 个 solve_tril case 另有 Mixed/F4。
每条路线检查计算结果，并对模型预测耗时进行默认 ±10% 误差验收。
任何一条失败，单 case 汇总即 FAIL；已失败路线之后未运行的路线会被 audit 标为缺失。
这里没有“Auto 必须是实测最快路线”的断言。

## PR 矩阵：运行全部 100 个 case

```bash
python costmodel_tests/run.py matrix --device 0

# 先跑 pytest 全集，再跑矩阵并审计；任一步失败，最终均为 FAIL
python costmodel_tests/run.py all --device 0
```

完整矩阵预期有 508 条路线记录（100 × 5 + 8 条 Mixed）。原 PR runner 默认只跑 Auto，
统一入口已经自动添加 `--include-explicit`，并在运行后调用 audit。
矩阵默认每个路线使用 3 个独立计时进程，取 profiler 平均耗时的中位样本。

显式指定环境位置：

```bash
python costmodel_tests/run.py matrix --device 0 \
  --profile /path/to/ascend_950pr_simd_simt_v1.json \
  --expected-bishengir /path/to/bin/bishengir-compile \
  --triton-root /path/to/compiler/triton-ascend \
  --npuir-root /path/to/AscendNPU-IR
```

`--triton-root`、`--npuir-root` 只用于记录被测编译器源码状态，不负责安装或切换 Python 包。
默认使用本仓目录；外部安装的编译器需要按上例指定源码路径。
`--profile` 同时用于编译器选路与 audit，防止模型和验收读取不同 profile。

## 输出与判定

统一入口默认输出到根目录的 `costmodel_test_results/<suite>_<时间>/`，也可用
`--output-root /new/empty/directory` 指定；拒绝覆盖非空目录。

```text
summary.json                       # 统一入口最终 PASS/FAIL 和各步骤退出码
pytest.xml                         # pytest JUnit 结果（执行 pytest 时）
matrix/
  matrix_summary.json              # 矩阵运行记录
  generalization_matrix.json       # 此次清单快照；单 case 时只有一项
  costmodel_profile.json           # profile 快照
  generalization_audit.md          # 各路线预测/实测/误差/PASS/FAIL
  generalization_audit.json        # all_pass 和 failure_reasons
  single_case_summary.json         # 仅单 case
  <case>__<route>_f<factor>/
    evidence.json                  # correct、profiler、编译/运行退出码等
    route.json                     # Auto 的候选与最终选择
    stdout.log / stderr.log
```

直接运行单 case 脚本时，默认输出到该脚本旁的 `single_case_results/`，
并额外生成 `runner.log`、`audit.log`。日志、trace、dump、结果默认不提交 Git。
审计判据：运行成功、`correct=true`、合法预测与实测存在，且
`abs(predicted_us / measured_us - 1) <= 0.10`。
profiler 数据优先；没有 profiler 时原审计脚本会退回 Event median，正式复现应保留 profiler。
只看到 runner 返回 0 不代表预测误差通过，必须检查最终 audit。

## 原 PR 脚本索引

目录：`third_party/ascend/unittest/costmodel_ut/costmodel_generalization/`。

| 脚本 | 用途 |
|---|---|
| `run_generalization_matrix.py` | case/family/route/shard 筛选，运行与证据收集 |
| `audit_generalization_matrix.py` | 逐路线检查预测误差，要求传入 manifest 的全部规定路线存在 |
| `run_generalization_shard.sh` | 按设备/分片运行完整路线；可设 `COSTMODEL_ENV_SCRIPT` 加载自己的环境 |
| `merge_generalization_runs.py` | 合并完整矩阵的多个分片结果 |
| `run_generalization_gates.sh` | 合并、合同自检和跨族残差诊断，适用于完整矩阵，不适用于单 case |
| `validate_generalization_contract.py` | 校验 profile、训练/留出族约束，不能用单 case 清单代替完整矩阵 |
| `fit_general_route_residual.py` | 离线模型残差诊断，不自动更改生产 profile |
| `collect_backend_evidence.py` | 隔离缓存/独立进程，保存 route、日志、后端资源证据 |
| `profile_utils.py` | NPU profiler 采集和 CSV 计时读取 |

`llvm-objdump` 在 PATH 时自动用于反汇编；也可直接给 collector 的 `--objdump` 指定。
没有该工具时不采集反汇编，不影响一般 profiler 计时，但不能据此声称验证了 spill。
原 `GENERAL_COSTMODEL_CONTRACT.md` 保留原文历史路径；实际操作以本说明为准。

## 后续增加 case

原始 pytest 测试放在 `pytest_ut/`；要进入 CostModel 性能验收，还需添加支持 route、factor、
report、profile-dir 和正确性输出的 benchmark，并在 `generalization_matrix.json` 登记
唯一 name、family、split、driver、arguments。记录原始文件和 commit，禁止按名字硬编码模型。
现有 `generalization_matrix_v140_all_100.json` 是历史快照，新增用例不要修改该快照。
新增后的总数和路线数以当前 manifest 为准。

## 本次验证范围

本分支的整理在 Windows 环境完成。验证 Python 语法、100-case 清单、驱动存在性、
main 的 pytest 文件完整保留、命令预览和模拟证据的真实 audit；没有执行 NPU 精度或性能测试。

无需 NPU 的测试资料自检：

```bash
python -m unittest discover -s costmodel_tests -p test_offline.py -v
```
