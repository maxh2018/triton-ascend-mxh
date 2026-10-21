# CostModel 测试使用说明

**选择清单 → 运行用例 → 检查精度和耗时。** 默认只运行 CI 的原始 100 个 case。
实际运行需准备兼容的 NPU 编译环境；查看清单和 `--dry-run` 不需要 NPU。

[脚本用法](#1-脚本用法) · [运行测试](#2-运行测试) · [结果与验收](#3-结果与验收)

## 1. 脚本用法

### 清单怎么选

**显式清单优先，分类只提供默认清单。**

1. 指定 `--manifest`：完全按该清单执行，`--category` 不参与选择，也不限制用例目录。
2. 未指定 `--manifest`：读取 `cases/<category>/manifest.json`。
3. 两者都未指定：读取 `cases/ci/manifest.json`（原始 100 个 case）。

显式清单不存在或格式错误时直接报错，不会回退到其他清单。以下是未指定 `--manifest` 时的默认值：

| 分类 | 默认清单 | 内容 |
|---|---|---|
| `ci`（默认） | [cases/ci/manifest.json](../cases/ci/manifest.json) | 原始 PR 100 个 case |
| `personal` | `cases/personal/manifest.json` | 已登记的个人用例，初始为空 |
| `operator_repos` | `cases/operator_repos/manifest.json` | vLLM、SGLang 等算子仓用例，初始为空 |

CI 另外保存 [manifest_all.json](../cases/ci/manifest_all.json)：100 个 benchmark 配置 + CI 目录下的 375 个测试文件。
**这份全量清单尚未验证，不会默认运行，必须用 `--manifest` 显式选择。**
测试文件中的函数和参数化节点在运行时展开，475 是清单条目数，不是算子数量。

### 命令语法

在**仓库根目录**执行：

```text
python costmodel_tests/run.py [run | list] [选项]
```

`run` 可以省略；`list` 只查看清单，不运行测试。不再区分 `matrix`、`pytest`、`all` 命令。

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--manifest <JSON路径>` | 未指定时使用分类默认值 | 最高优先级；相对路径以当前目录为准，也支持绝对路径 |
| `--category <分类>` | `ci` | 仅在没有 `--manifest` 时选择默认清单，取值 `ci`、`personal`、`operator_repos` |
| `--case <名称>` | 整份清单 | 只运行指定条目，可重复填写；名称来自清单的 `name` |
| `--device <卡号>` | 继承环境 | 使用指定 NPU，例如 `--device 0` |
| `--devices <卡号列表>` | 无 | 多卡设备池，例如 `0,1`；与 `--device` 互斥 |
| `--jobs <数量>` | `1` | 同时运行的清单条目数 |
| `--retries <次数>` | `3` | 失败后额外重试次数；默认最多尝试 4 次，`0` 不重试 |
| `--output-root <目录>` | 自动生成 | 结果目录，必须是新目录或空目录 |
| `--dry-run` | 关闭 | 只打印计划，不运行测试、不创建结果目录 |

**清单中的 `driver` 相对于清单所在目录，也支持绝对路径。** 一份清单可同时引用 CI、个人和算子仓用例；分类用于存放文件，清单决定本次执行哪些用例。
字段说明和最小例子见 [自定义用例提交规范](CUSTOM_CASES.md)。

<details>
<summary>运行环境与进阶参数</summary>

在具有空闲 Ascend NPU 的 Linux 服务器上，先加载 CANN、Triton-Ascend、torch_npu 和 bishengir 环境。
本分支只保存测试资料，需另外安装支持 PR #2528 选路参数和报告格式的兼容编译器。
基础 Python 依赖：`pytest`、`pytest-xdist`、`torch`、`torch_npu`、`triton`；部分用例另有依赖。

```bash
unset TRITON_ASCEND_COMPILE_MODE
python costmodel_tests/run.py --help
```

| 选项 | 用途 |
|---|---|
| `--profile <JSON路径>` | 编译和审计使用同一份 profile；默认对应 950PR/A5，其他芯片需指定兼容文件 |
| `--expected-bishengir <文件路径>` | 指定预期的 `bishengir-compile`；默认从 PATH 查找 |
| `--triton-root <目录>` | 记录被测 Triton-Ascend 源码状态，不安装或切换 Python 包 |
| `--npuir-root <目录>` | 记录被测 NPU-IR 源码状态，不切换工具链 |
| `--validation-scope auto / all` | 默认只验收 Auto；`all` 为 benchmark 增加规定的显式路线，pytest 仍验收 Auto |

pytest 通过公共适配器保留原断言并捕获 JIT kernel；每种启动配置计时一次，不额外重放 kernel，
以免改变原子操作或原地更新结果。benchmark 沿用原 PR 的计时协议。
一个 pytest 条目内不支持 `-n` 并行；用 `--devices`、`--jobs` 在条目之间并发。

</details>

## 2. 运行测试

### 运行自己的清单（优先）

```bash
python costmodel_tests/run.py --manifest my_tests.json --device 0
```

无需指定 `--category`。可先用仓库自带的 [跨目录示例清单](mixed_manifest.example.json) 查看组合：

```bash
python costmodel_tests/run.py list --manifest costmodel_tests/mixed_manifest.example.json
```

这份示例同时引用个人向量加法 benchmark 和 CI 的 pytest 向量加法文件，尚未进行 NPU 验证。
将 `list` 去掉即可按清单执行；清单可以保存在分类目录之外。

### 默认运行：CI 原始 100 个 case

```bash
python costmodel_tests/run.py --device 0
```

等价于显式选择 `--manifest cases/ci/manifest.json`。

### 查看或只运行指定 case

```bash
# 查看当前默认清单
python costmodel_tests/run.py list

# 只运行一个；--case 填清单名称，不填 .py 文件名
python costmodel_tests/run.py --case softmax_original --device 0

# 选几个一起运行
python costmodel_tests/run.py --case softmax_original --case muls_add_n2097152 --device 0
```

### 显式运行 CI 全量清单（未验证）

```bash
# 先查看，不执行
python costmodel_tests/run.py list --manifest cases/ci/manifest_all.json

# 先选一个向量加法测试验证环境
python costmodel_tests/run.py --manifest cases/ci/manifest_all.json \
  --case pytest_test_01_vector_add --device 0

# 确认后运行整份清单
python costmodel_tests/run.py --manifest cases/ci/manifest_all.json --device 0
```

没有实际运行 kernel 的源码、接口、编译类测试会报告 `UNSUPPORTED`，并使严格验收失败；
缺少预测或计时同样不能通过。原有跳过项仍记为 `SKIP`，不计入通过数。
全量清单保留这些条目以便后续验证和整理，不表示所有条目已经适配完成。

### 个人与算子仓用例

```bash
# 使用个人目录的默认清单
python costmodel_tests/run.py --category personal --device 0

# 使用算子仓目录的默认清单
python costmodel_tests/run.py --category operator_repos --device 0

# 使用仓库提供的个人向量加法示例清单
python costmodel_tests/run.py --manifest cases/personal/examples/custom_manifest.example.json --device 0
```

默认清单为空时，可用 `list` 查看；登记用例后才能运行。
个人算子放在 `cases/personal/<姓名>/`，上传方法见 [提交规范](CUSTOM_CASES.md)。

### 并发与失败重试

```bash
# 两张卡并发；每张卡同一时间只运行一个条目
python costmodel_tests/run.py --devices 0,1 --jobs 2 --retries 3

# 单卡串行，关闭重试
python costmodel_tests/run.py --device 0 --retries 0
```

只重试失败用例；已通过的清单条目不会重跑。
同一 pytest 文件内，也只重试失败的参数化节点，已通过或跳过的节点保留结果。
若失败发生在收集阶段、尚无节点结果，则重试该文件。无法定位失败节点且已有通过结果时，不会全量重放该文件。

`--devices` 的卡数不得少于 `--jobs`，卡号不能重复，设备型号需与 profile 匹配。
同卡并发会干扰耗时测量，验收建议单卡串行或多卡隔离。重试不支持从旧目录断点续跑。

## 3. 结果与验收

### 结果在哪里

终端打印 `Output: <目录>`，默认保存到 `costmodel_test_results/run_<时间>/`。
所有清单使用相同的结果入口：

| 文件 | 用途 |
|---|---|
| `summary.json` | **整次运行的 PASS/FAIL**、实际清单、选择来源和退出码 |
| `report.md` | **优先阅读**：每个条目的精度、耗时、尝试次数，以及预测/实测/误差 |
| `report.json` | 完整结果，供程序读取 |
| `batch_summary.json` | `failed_cases`、每次日志路径和最终尝试目录 |
| `batch_summary.partial.json` | 执行中的完成进度 |
| `manifest.json`、`costmodel_profile.json` | 本次清单和 profile 快照 |
| `cases/<名称>/attempt_01.log` | 首次日志；重试依次生成 `attempt_02.log` 等 |

`summary.json` 的 `manifest_source` 为 `explicit`（显式指定）或 `category_default`（分类默认值）。显式清单的 `category` 为 `null`，表示这次运行不归属于某个分类目录。

<details>
<summary>单条用例的原始证据</summary>

每次尝试的文件放在 `cases/<名称>/attempt_XX/`：

- benchmark：`runner.log`、`audit.log`、`generalization_audit.json`，以及各路线的 `route.json`、`evidence.json`。
- pytest：`pytest.xml` 和 `pytest_costmodel/` 下的报告、逐节点 `result.json`、选路报告和 profiler 数据。

pytest 跨重试合并后的节点结果另存为 `cases/<名称>/pytest_summary.json`。
用 `batch_summary.json` 中的 `final_output_root`、`attempts[].log` 找最后一次尝试和日志。

</details>

### 通过标准

1. **精度通过**：运行成功，计算结果满足参考结果和该用例的精度容差。
2. **耗时通过**：Auto 所选路线的预测、实测耗时齐全，绝对相对误差 ≤ 10%。

```text
relative_error = predicted_us / measured_us - 1
通过条件：abs(relative_error) <= 0.10
```

例如实测 `10 μs`、预测 `10.8 μs`，误差 `+8%`，通过；预测 `11.5 μs` 则为 `+15%`，失败。
**10% 是耗时预测误差，不是精度误差。** 不要求 Auto 是实测最快路线。

整次运行要求所有所选条目通过，退出码为 `0`，`summary.json` 的 `status` 为 `PASS`。
`UNSUPPORTED` 不能算通过；一个用例有多个被测 kernel 配置时，所有采集配置都要通过。

### 失败时怎么看

1. 看 `report.md`，确认是精度、耗时还是 `UNSUPPORTED`。
2. 在 `batch_summary.json` 查 `failed_cases`，按日志路径查看最后一次失败。
3. 精度断言与编译异常看运行日志或 `pytest.xml`；耗时问题看原始报告中的 `predicted_us`、`measured_us`、`relative_error` 和 `failure_reasons`。

---

新增 CI 文件后可执行 `python costmodel_tests/update_ci_manifests.py` 更新两份清单；此操作只生成清单，不进行验证，且不会更改默认选择。
用例来源见 [sources.json](sources.json)，依赖见 [dependencies.json](dependencies.json)。
本地仅做 CPU 回归与格式校验，未执行 NPU 精度或性能测试；资料自检：`python -m unittest discover -s costmodel_tests -p 'test_*.py' -v`。
