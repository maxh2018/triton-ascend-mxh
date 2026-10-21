# CostModel 测试用例与脚本使用说明

本分支 `costmodel/test-cases` 用于保存和持续扩充 CostModel 测试资料。

## 收录范围和来源

所有用例统一放在 `cases/`，按来源分三类：

```text
cases/
  ci/                       # 现有 CI 用例，内部保留原相对目录层级
  personal/
    manifest.json           # 个人用例清单（初始为空）
    <姓名>/                 # 个人算子上传位置
    examples/               # 向量加法示例，供复制
  operator_repos/
    manifest.json           # 算子仓用例清单（初始为空）
    vllm/<算子名>/          # vLLM 来源用例
    sglang/<算子名>/        # SGLang 来源用例
```

### 第一次使用：先选用例，再选运行方式

`costmodel_tests/run.py` 是统一运行入口。下面的命令都在**仓库根目录**执行；
查看用例不需要 NPU，实际运行前需准备下文的 NPU 编译环境。

**1. 选哪一类用例？**

`--category` 表示用例来源，不填时默认选 `ci`。

| 参数 | 会读取哪份用例清单 | 当前内容 |
|---|---|---|
| `--category ci` | `cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/generalization_matrix.json` | 已登记的 PR 100 个 case |
| `--category personal` | `cases/personal/manifest.json` | 个人用例，初始为空 |
| `--category operator_repos` | `cases/operator_repos/manifest.json` | vLLM、SGLang 等算子仓用例，初始为空 |

“清单”是一个 JSON 文件，记录每个 case 的名字、要执行的 Python 脚本和输入参数。
**只把脚本放进目录还不够，必须在清单中添加配置，运行器才知道如何执行它。**
例如个人脚本放在 `cases/personal/zhangsan/bench_my_add.py`，
就在 `cases/personal/manifest.json` 中登记，`driver` 填 `zhangsan/bench_my_add.py`。
完整配置示例见 [自定义用例提交规范](CUSTOM_CASES.md)。

**2. 查看、运行一个，还是运行整份清单？**

| 命令 | 作用 |
|---|---|
| `list` | 查看所选清单的 case 名称，不执行测试 |
| `case --case <名称>` | 只运行清单中指定名字的一个 case；名称来自 JSON 的 `name`，不是 Python 文件名 |
| `matrix` | 运行所选清单中的全部 case |
| `pytest` | 运行 CI 目录下原始 `pytest_ut` 测试集，不读取上述 case 清单 |
| `all` | 先运行 CI 原始 pytest 测试集，再运行所选清单的 CostModel 测试；仅支持 CI 分类 |

个人和算子仓用例使用 `list`、`case`、`matrix`。
清单为空（`"cases": []`）时，`list` 会显示 0 个 case；此时没有可运行的测试，
应先登记用例。仓库提供了一份已登记一个向量加法 case 的示例清单，可用于首次尝试。

```bash
# 查看已有的 CI 100 个 case
python costmodel_tests/run.py list

# 从中选一个运行：softmax_original 是清单里的 case 名
python costmodel_tests/run.py case --case softmax_original --device 0

# 运行 CI 清单中的全部 100 个 case
python costmodel_tests/run.py matrix --device 0

# 查看自己登记的个人用例；未登记时显示 0 个
python costmodel_tests/run.py list --category personal
```

**3. 不想使用分类默认清单，怎么运行另一份清单？**

加 `--manifest <JSON路径>`，本次就读取指定文件，代替表格中的默认清单；
不会修改任何清单文件。比如，运行仓库自带的个人向量加法示例：

```bash
python costmodel_tests/run.py case --category personal \
  --manifest cases/personal/examples/custom_manifest.example.json \
  --case custom_add_f32_n1024 --device 0
```

个人用例登记到默认清单后，则不需要 `--manifest`，例如：

```bash
# 以下命令需先登记名为 zhangsan_add_f32_n1024 的 case
python costmodel_tests/run.py case --category personal \
  --case zhangsan_add_f32_n1024 --device 0

# 批跑个人默认清单中的全部 case
python costmodel_tests/run.py matrix --category personal --device 0
```

**编写测试脚本时，如何复用计时函数？**

仓库已有 `profile_utils.py`，其中的 `profile_kernel` 用来采集 NPU 实际运行耗时。
通过上述统一入口运行时，你的个人或算子仓脚本可以直接写：

```python
from profile_utils import profile_kernel
```

运行器会设置 Python 的模块搜索路径，所以不需要把 `profile_utils.py` 复制到自己的目录。
这只说明如何导入计时函数；完整调用方式请参考 [向量加法示例](../cases/personal/examples/bench_custom_add_example.py)。
直接用 `python 你的脚本.py` 启动时，不保证有这个搜索路径，建议使用统一入口运行。

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
python cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/run_generalization_case.py \
  softmax_original --device 0
```

case 参数来自矩阵的 `name`，不能填 `bench_*.py` 文件名。
默认 `--validation-scope auto`，与原 PR 的 100-case Auto 结果表口径一致：
只运行 Auto 最终选择的路线，检查该路线的计算结果和 ±10% 耗时预测误差。
Auto 可以选择 SIMD、SIMT 或 Mixed 及对应 factor；不会单独强制执行其他候选路线。
需要完整路线验收时设置 `--validation-scope all`，才会额外运行 SIMD、SIMT F1/F2/F4，
以及清单中的额外 Mixed 配置，并要求全部规定路线通过。
这里没有“Auto 必须是实测最快路线”的断言。
统一入口 `run.py case` 会在单次尝试失败后重试整个 case，默认额外重试 3 次。
直接调用 `run_generalization_case.py` 仍然只执行一次；重试由 batch 调度层负责。

## PR 矩阵：运行全部 100 个 case

```bash
python costmodel_tests/run.py matrix --device 0

# 先跑 pytest 全集，再跑矩阵并审计；任一步失败，最终均为 FAIL
python costmodel_tests/run.py all --device 0
```

默认矩阵预期有 **100 条 Auto 路线记录**，每个 case 一条，与原 PR 结果表范围一致。
完整模式 `--validation-scope all` 才有 508 条路线记录（100 × 5 + 8 条 Mixed），
此时单次执行添加 `--include-explicit`。原 PR 的 audit 本身要求完整路线，
本分支新增了 Auto-only 审计范围，并在 runner、batch 和 audit 之间统一传递。
默认 Auto-only 不会因为未选中路线的预测误差或缺失而失败。

```bash
# 可选：扩展为完整路线验收
python costmodel_tests/run.py matrix --device 0 --validation-scope all
```
矩阵默认每个路线使用 3 个独立计时进程，取 profiler 平均耗时的中位样本。

## 并发与失败重试

`--jobs` 控制同时运行的 **case 数**，默认 1；同一 case 内的各条路线仍串行。
`--retries` 是首次失败后的**额外重试次数**，默认 3，即一个 case 最多运行 4 次。
设置 0 则不重试。编译/运行失败、精度失败、缺少证据、预测误差 audit 失败均可触发重试。
只重试失败的 case：默认重新执行 Auto；完整模式重新执行该 case 的全部规定路线。
已经通过的其他 case 不会再运行。

```bash
# 单卡上同时运行 4 个 case，失败后额外重试 3 次
python costmodel_tests/run.py matrix --device 0 --jobs 4 --retries 3

# 四张卡并发：每张卡同一时间只运行一个 case，减少测时相互干扰
python costmodel_tests/run.py matrix --devices 0,1,2,3 --jobs 4 --retries 3

# 禁用失败重试
python costmodel_tests/run.py matrix --device 0 --jobs 1 --retries 0

# 单 case 失败后额外重试 5 次
python costmodel_tests/run.py case --case softmax_original --device 0 --retries 5
```

`--device` 和 `--devices` 互斥；`--devices` 需要至少提供 `--jobs` 张互不重复的卡。
同一张 NPU 并发执行多个 case 会产生资源竞争，可能改变 profiler 耗时和 ±10% 审计结果。
因此精度/预测误差验收建议使用多卡隔离并发，或保持 `--jobs 1`。
设备池中的卡应具有与 profile 一致的目标型号。
这些并发和重试选项作用于 100-case 矩阵（`matrix`、`case`、`all` 的矩阵阶段），
不改变原始 pytest 的执行策略；pytest 参数继续使用 `--pytest-args`。

仅批跑指定的几个 case，可直接使用新 batch 脚本：

```bash
python cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/run_generalization_batch.py \
  --case softmax_original --case atomic_discrete_i32_p1_b256 \
  --devices 0,1 --jobs 2 --retries 3 --output-root /tmp/costmodel_selected_new
```

终端显示每次 START/PASS/FAIL/RETRY，重试用尽后仍继续完成其他 case，最终返回非零退出码。
该功能是当前一次运行内的自动重试，不是对已有结果目录的断点续跑。

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
  batch_summary.json               # case 状态、尝试次数、设备、每次日志位置、最终失败列表
  batch_summary.partial.json       # 每完成一个 case 更新一次的进度
  generalization_matrix.json       # 此次清单快照；单 case 时只有一项
  costmodel_profile.json           # profile 快照
  generalization_audit.md          # 各路线预测/实测/误差/PASS/FAIL
  generalization_audit.json        # all_pass 和 failure_reasons
  cases/<case>/
    attempt_01.log                 # 首次运行终端输出
    attempt_01/                    # 首次运行的完整证据
      single_case_summary.json
      generalization_audit.json
      <case>__<route>_f<factor>/
        evidence.json              # correct、profiler、编译/运行退出码等
        route.json                 # Auto 的候选与最终选择
        stdout.log / stderr.log
    attempt_02.log                 # 失败重试时才生成
    attempt_02/
```

直接运行单 case 脚本时，默认输出到该脚本旁的 `single_case_results/`，
并额外生成 `runner.log`、`audit.log`。日志、trace、dump、结果默认不提交 Git。
batch 汇总只引用各 case 的最后一次尝试，之前的失败日志始终保留；最终 audit 使用同一
`validation_scope`（默认 `auto`）。`batch_summary.json` 的 `failed_cases` 列出重试用尽仍失败的 case。
batch 的 `matrix_summary.json` 供 audit 读取；原 PR 的 shard 合并/跨族 gates 仍应使用原始
`run_generalization_matrix.py`/`run_generalization_shard.sh` 的完整元数据输出。
审计判据：运行成功、`correct=true`、合法预测与实测存在，且
`abs(predicted_us / measured_us - 1) <= 0.10`。
profiler 数据优先；没有 profiler 时原审计脚本会退回 Event median，正式复现应保留 profiler。
只看到 runner 返回 0 不代表预测误差通过，必须检查最终 audit。

## 后续增加 case

按 [自定义测试用例提交规范](CUSTOM_CASES.md) 提交：**一个 Python 脚本 + 一条清单配置**。
文档包含文件位置、必需参数、结果格式和运行命令。
个人 case 登记在 `cases/personal/manifest.json`，使用 `--category personal` 选择；
算子仓 case 登记在 `cases/operator_repos/manifest.json`，使用 `--category operator_repos` 选择。
也可用 `--manifest` 指定独立清单，原 PR 的
`generalization_matrix.json` 和 `generalization_matrix_v140_all_100.json` 保持不变。

## 本次验证范围

本分支的整理在 Windows 环境完成。验证 Python 语法、100-case 清单、驱动存在性、
main 的 pytest 文件完整保留、命令预览和模拟证据的真实 audit；没有执行 NPU 精度或性能测试。

无需 NPU 的测试资料自检：

```bash
python -m unittest discover -s costmodel_tests -p 'test_*.py' -v
```
