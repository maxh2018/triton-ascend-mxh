# CostModel 测试使用说明

统一入口：`costmodel_tests/run.py`。默认测试集包含 **CI 原始 pytest 测试 + PR #2528 的 100 个 CostModel case**。

[脚本用法](#1-脚本用法) · [运行测试](#2-运行测试) · [结果与验收](#3-结果与验收)

## 1. 脚本用法

### 运行前准备

在已加载 CANN、Triton-Ascend、torch_npu 和 bishengir 环境的 **Ascend NPU Linux 服务器**上运行。
本分支只保存测试资料，编译器需单独安装，并支持 PR #2528 的选路参数和报告格式。
默认 profile 对应 950PR/A5；其他芯片需用 `--profile` 指定兼容文件。

以下命令均从**仓库根目录**执行。`--device 0` 表示使用卡 0，可换成自己的空闲卡号。
运行前取消会覆盖选路的环境变量：

```bash
unset TRITON_ASCEND_COMPILE_MODE
python costmodel_tests/run.py --help
```

### 命令语法

```text
python costmodel_tests/run.py <命令> [选项]
```

| 命令 | 执行内容 |
|---|---|
| `list` | 查看清单中的 case 名称，不执行测试 |
| `case` | 运行一个 case，必须带 `--case <名称>` |
| `matrix` | 运行所选清单中的全部 CostModel case；默认是原始 100 个 |
| `pytest` | 运行 CI 原始 `pytest_ut` 测试集 |
| `all` | 先运行 CI pytest，再运行 CostModel 清单；默认覆盖上述两套测试 |

`pytest`、`all` 仅支持 CI 分类。`all` 不会自动遍历个人和算子仓目录。

### 常用选项

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--case <名称>` | 无 | 单 case 名称，来自清单的 `name`，不能填 `.py` 文件名 |
| `--category <分类>` | `ci` | 选择 `ci`、`personal` 或 `operator_repos` |
| `--manifest <JSON路径>` | 分类默认清单 | 本次改用指定清单，不修改原文件 |
| `--device <卡号>` | 继承环境 | 使用指定 NPU，例如 `--device 0` |
| `--devices <卡号列表>` | 无 | 多卡设备池，例如 `--devices 0,1`；与 `--device` 互斥 |
| `--jobs <数量>` | `1` | 同时运行的 CostModel case 数 |
| `--retries <次数>` | `3` | 失败后额外重试次数；`0` 不重试，默认最多尝试 4 次 |
| `--validation-scope <范围>` | `auto` | `auto` 只验收 Auto 所选路线；`all` 验收全部规定路线 |
| `--output-root <目录>` | 自动生成 | 结果保存位置；必须是新目录或空目录 |
| `--dry-run` | 关闭 | 只打印执行命令，不运行测试、不创建结果目录 |
| `--pytest-args ...` | 无 | 透传 pytest 参数，必须放在命令最后 |

并发和重试只作用于 CostModel case，不改变 pytest 的执行方式。
`--devices` 提供的卡数不得少于 `--jobs`，且卡号不能重复。

<details>
<summary>编译环境与计时参数</summary>

| 选项 | 用途 |
|---|---|
| `--profile <JSON路径>` | 同时指定编译器选路和审计使用的 CostModel profile |
| `--expected-bishengir <文件路径>` | 指定预期的 `bishengir-compile`；默认从 PATH 查找 |
| `--triton-root <目录>` | 记录被测 Triton-Ascend 源码状态，不安装或切换 Python 包 |
| `--npuir-root <目录>` | 记录被测 NPU-IR 源码状态，不切换工具链 |

矩阵默认每条路线用 3 个独立进程计时，取 profiler 平均耗时的中位样本。
如需调整预热、重复次数，可使用底层 [单 case 脚本](../cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/run_generalization_case.py)
的 `--warmup`、`--repeats`、`--measurement-processes`。直接调用该脚本不会自动重试。

</details>

## 2. 运行测试

### 完整 CI 测试：pytest + 原始 100 个 case

```bash
python costmodel_tests/run.py all --device 0
```

默认先跑 CI 原始 pytest，再跑 100 个 CostModel case。两部分都通过，总结果才是 `PASS`。

### 只跑原始 100 个 case

```bash
python costmodel_tests/run.py matrix --device 0
```

使用仓库内保留的 PR 100-case 清单，默认只验收 Auto 最终选择的路线。
如需先确认 case 名称或执行计划：

```bash
python costmodel_tests/run.py list
python costmodel_tests/run.py matrix --device 0 --dry-run
```

### 单个 case / 原始 pytest

```bash
# 单个 CostModel case
python costmodel_tests/run.py case --case softmax_original --device 0

# 只跑 CI 原始 pytest
python costmodel_tests/run.py pytest --device 0

# 筛选 pytest 中名称包含 atomic 的测试
python costmodel_tests/run.py pytest --device 0 --pytest-args -k atomic
```

### 并发与失败重试

```bash
# 两张卡并发，每卡一个 case；失败后额外重试 3 次
python costmodel_tests/run.py matrix --devices 0,1 --jobs 2 --retries 3

# 单卡串行，禁用重试
python costmodel_tests/run.py matrix --device 0 --retries 0
```

**只重试失败的 case，已通过的 case 不会重跑。** 每次尝试的日志独立保存，最终汇总使用最后一次尝试。
同卡并发会干扰耗时测量，预测误差验收建议单卡串行或多卡隔离；设备型号需与 profile 匹配。
重试仅针对本次运行，不是从旧结果目录断点续跑。

### 个人用例与算子仓用例

“清单”是记录 case 名称、脚本路径和输入参数的 JSON 文件。脚本放入目录后，还需登记到清单才能运行。

| 分类 | 脚本保存位置 | 默认清单 |
|---|---|---|
| `ci` | `cases/ci/` | [PR 原始 100-case 清单](../cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/generalization_matrix.json) |
| `personal` | `cases/personal/<姓名>/` | `cases/personal/manifest.json` |
| `operator_repos` | `cases/operator_repos/vllm/`、`sglang/` 等 | `cases/operator_repos/manifest.json` |

```bash
# 查看、批跑已登记的个人用例；算子仓将 personal 换为 operator_repos
python costmodel_tests/run.py list --category personal
python costmodel_tests/run.py matrix --category personal --device 0

# 运行仓库自带的向量加法示例，无需先修改个人清单
python costmodel_tests/run.py case --category personal \
  --manifest cases/personal/examples/custom_manifest.example.json \
  --case custom_add_f32_n1024 --device 0
```

个人和算子仓默认清单初始为空，需先登记再运行。
上传位置、脚本接口及 JSON 每个字段的含义见 [自定义用例提交规范](CUSTOM_CASES.md)。

## 3. 结果与验收

### 结果在哪里

终端会打印 `Output: <结果目录>`。默认路径是：

```text
costmodel_test_results/<命令>_<时间>/
```

可加 `--output-root /tmp/my_costmodel_run` 指定新目录。仅运行 pytest 时不会生成 `matrix/`。

| 文件（相对于结果目录） | 看什么 |
|---|---|
| `summary.json` | **先看这里**：整次测试的 `status`，以及各步骤的退出码 |
| `pytest.xml` | 原始 pytest 的失败、报错、跳过记录；执行 pytest 时生成 |
| `matrix/generalization_audit.md` | **CostModel 主要报告**：每个 case 的预测耗时、实测耗时、误差、PASS/FAIL |
| `matrix/generalization_audit.json` | 审计原始字段，便于自动统计和排错 |
| `matrix/batch_summary.json` | 最终失败列表、每个 case 的尝试次数和日志路径 |
| `matrix/batch_summary.partial.json` | 运行中的完成进度 |
| `matrix/cases/<case>/attempt_01.log` | 首次尝试日志；重试依次生成 `attempt_02.log` 等 |

<details>
<summary>其他文件与原始证据</summary>

```text
matrix/
  generalization_matrix.json      # 本次执行的清单快照
  costmodel_profile.json          # 本次使用的 profile 快照
  matrix_summary.json             # 各 case 的执行记录，供审计读取
  cases/<case>/attempt_01/
    single_case_summary.json      # 本次尝试的结果
    runner.log / audit.log         # 运行与审计日志
    <case>__auto_f1/
      evidence.json               # 正确性、计时、运行退出码
      route.json                  # 编译器候选路线和最终选择
      stdout.log / stderr.log     # benchmark 原始输出
```

`auto_f1` 是 Auto 运行记录的名称，实际选中的路线和 factor 请查看 `route.json`。

</details>

### 怎么判断通过

| 测试类型 | 通过标准 |
|---|---|
| 原始 pytest | 按各测试自身的精度、编译或接口断言判定；不统一检查 CostModel ±10% 误差 |
| CostModel case（默认 `auto`） | Auto 所选路线运行成功、计算正确，预测与实测耗时齐全，且绝对相对误差 ≤ 10% |
| 整次运行 | 所有被执行的测试部分通过；最终退出码为 `0`，`summary.json` 的 `status` 为 `PASS` |

预测误差计算方式：

```text
relative_error = predicted_us / measured_us - 1
通过条件：abs(relative_error) <= 0.10
```

例如实测 `10 μs`、预测 `10.8 μs`，误差为 `+8%`，通过；预测 `11.5 μs` 则为 `+15%`，失败。
精度容差由各 case 自身的正确性断言规定，**10% 是耗时预测误差，不是精度误差**。
实测优先使用 profiler 耗时；审计支持 Event median 回退，正式复现应保留 profiler 数据。

默认与 PR 的 100-case Auto 结果表范围一致：100 个 case 对应 **100 条 Auto 记录**，
不要求未选中的路线也通过，也不检查 Auto 是否实测最快。
如需扩展到全部规定路线（原始清单共 508 条记录），执行：

```bash
python costmodel_tests/run.py matrix --device 0 --validation-scope all
```

这里的 `--validation-scope all` 扩大的是**路线验收范围**，与运行 pytest + 矩阵的 `all` 命令含义不同。

### 失败时怎么看

1. 打开 `summary.json`，确认失败的是 `pytest` 还是 `matrix` / `case` 步骤。
2. pytest 失败查看终端和 `pytest.xml`；CostModel 失败先看 `matrix/generalization_audit.md` 的 `Gate` 列。
3. 在 `matrix/batch_summary.json` 查 `failed_cases`；按对应 case 的 `final_output_root`、`attempts[].log` 找最后一次日志。

| 审计字段 | 含义 |
|---|---|
| `all_pass` | 审计是否全部通过 |
| `results[].predicted_us` / `measured_us` | 预测 / 实测耗时，单位微秒 |
| `results[].relative_error` | 相对误差；`0.08` 即 `+8%` |
| `results[].pass` | 该条记录是否通过 |
| `results[].failure_reasons` | 缺少报告、计算不正确、计时缺失等原因 |

纯粹超出 10% 时，`failure_reasons` 可能为空，应结合 `pass` 和 `relative_error` 判断；
只有运行脚本退出成功，也不能替代最终审计结果。

---

用例来源见 [sources.json](sources.json)，保留依赖见 [dependencies.json](dependencies.json)。
当前仓库整理与脚本校验在 Windows 上完成，未在本地执行 NPU 精度或性能测试。
无需 NPU 的资料自检：`python -m unittest discover -s costmodel_tests -p 'test_*.py' -v`。
