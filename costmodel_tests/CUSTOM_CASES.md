# 自定义测试用例提交规范

**准备一个 Python 测试脚本，在 JSON 清单中登记，再按清单运行。**

[运行示例](#1-先运行一个示例) · [上传位置](#2-文件放在哪里) · [清单填写](#3-清单怎么写) · [脚本接口](#4-脚本需要满足什么接口) · [运行与验收](#5-运行与查看结果)

## 1. 先运行一个示例

仓库提供一个向量加法 `x + y` 示例：

- [示例脚本](../cases/personal/examples/bench_custom_add_example.py)
- [示例清单：1 个 case](../cases/personal/examples/custom_manifest.example.json)

示例清单的字段含义：

| 字段 | 示例值 | 含义 |
|---|---|---|
| `cases` | `[{...}]` | 用例列表；每个对象代表一个测试配置 |
| `name` | `custom_add_f32_n1024` | 清单内唯一的用例名，`--case` 填这个值 |
| `runner` | `benchmark` | 可选，默认 `benchmark`；已有 pytest 文件填 `pytest` |
| `driver` | `bench_custom_add_example.py` | Python 脚本路径，相对于清单所在目录 |
| `arguments` | `["--n-elements", "1024"]` | 脚本参数，按顺序填写，所有元素均为字符串；这里表示 1024 个元素 |
| `family` | `elementwise` | 算子类别，用于分类统计；这里表示逐元素计算 |
| `split` | `holdout` | 可选，默认 `holdout`（留出验证）；`train` 表示模型开发用例，仅作为标签 |

加载兼容的 NPU 编译环境后，在**仓库根目录**执行：

```bash
python costmodel_tests/run.py --manifest cases/personal/examples/custom_manifest.example.json \
  --case custom_add_f32_n1024 --device 0
```

执行流程：**生成输入 → 运行 kernel → 检查精度 → 采集耗时 → 输出结果**。

添加自己的算子时，可以复制这两个文件，替换 kernel、参考结果和清单中的名字、参数。示例展示接口，不代表当前 profile 对该 shape 已实测通过。

## 2. 文件放在哪里

**个人算子放到 `cases/personal/<你的名字>/`，在 `cases/personal/manifest.json` 登记。**

```text
cases/personal/
  manifest.json                  # 个人用例清单，多人共用
  zhangsan/
    bench_custom_my_add.py        # kernel 和测试脚本
```

| 来源 | 存放目录 |
|---|---|
| 个人用例 | `cases/personal/<你的名字>/` |
| 算子仓用例，如 vLLM、SGLang | `cases/operator_repos/` |
| 已有 CI 用例 | `cases/ci/`；个人用例不要放到这里 |

提交时注意：

- **命名**：建议用字母、数字和下划线，清单中的用例名不能重复。
- **来源与依赖**：在脚本头部注明来源、额外依赖和适用芯片；输入尽量在脚本中生成。
- **辅助文件**：随脚本一起提交，按脚本所在位置读取；不要提交编译缓存、profiler 数据和日志。
- **共享 helper**：运行器已提供导入路径，可直接使用 `from profile_utils import profile_kernel`，无需复制文件。

## 3. 清单怎么写

在 `cases/personal/manifest.json` 的 `cases` 列表中追加条目，**保留已有用例和原 PR 的 100-case 清单**：

```json
{
  "cases": [
    {
      "name": "custom_my_add_f32_n1024",
      "runner": "benchmark",
      "driver": "zhangsan/bench_custom_my_add.py",
      "arguments": ["--n-elements", "1024"],
      "family": "elementwise",
      "split": "holdout"
    }
  ]
}
```

填写时记住三点：

- **路径**：`driver` 相对于清单所在目录；提交仓库时使用相对路径。本机临时清单也支持绝对路径。
- **参数**：`arguments` 只填脚本自定义参数。benchmark 的 `route`、`factor`、`report`、`profile-dir`、`warmup`、`repeats` 由运行器追加。
- **配置**：同一 kernel 的不同 shape 可以分别登记，使用不同的 `name` 和 `arguments`。

同一清单可以用 `../` 引用不同目录的用例，参考 [混合清单示例](mixed_manifest.example.json)。显式 `--manifest` 优先于 `--category`。

## 4. 脚本需要满足什么接口

### 新写 benchmark 脚本

建议复制 [向量加法示例](../cases/personal/examples/bench_custom_add_example.py) 或 [现有 benchmark 模板](../cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/bench_muls_add_costmodel.py)，保留参数解析、选路、精度检查、计时和 JSON 输出部分。

**一个 benchmark case 对应一个被测 kernel。** 脚本需接受以下公共参数；自己的 shape、dtype 等参数可以另外添加。

| 参数 | 要求 |
|---|---|
| `--route` | 支持 `auto`、`simd`、`simt_only` |
| `--factor` | 支持 `1`、`2`、`4`，传给 kernel 的 `superblock_factor` |
| `--warmup`、`--repeats` | 预热和重复计时次数 |
| `--report` | Auto 模式下编译器选路报告的输出路径 |
| `--profile-dir` | 可选；提供时采集 NPU profiler，不提供时也应能运行 |

#### 编译与选路

- 按 `--route` 和 `--factor` 编译运行，不要固定一种路线。
- Auto 设置 `compile_mode="simd_simt"`、`auto_simt_scope_mode="auto"`，将 `--report` 传给 `auto_simt_scope_dump`；显式路线关闭自动选路。
- `logical_program_count_hint` 填真实逻辑 program 数，物理核数从设备属性获取。
- 选路报告由兼容编译器生成，不要自行填写预测耗时。暂不要求增加 Mixed 等额外模式。

#### 精度与计时

- 使用参考结果检查输出，例如 `torch.testing.assert_close()`；失败时抛异常或返回非零退出码。
- 各路线保持相同计算语义，不能跳过不支持的路线后报告成功。
- 提供 `--profile-dir` 时，使用 `profile_kernel(launch, args.profile_dir, kernel_name)` 采集实测耗时。

#### 结果输出

最后向标准输出打印一个 JSON 对象，至少包含：

```python
print(json.dumps({
    "case": "custom_my_add_f32_n1024",
    "correct": True,        # 仅在真实精度检查通过后填写 True
    "profiler": profiler,   # profile_kernel() 返回值；未采集时为 None
}))
```

`profiler` 包含实测耗时 `average_us`，单位为微秒。可以在此前打印日志，最终结果后不要再打印其他 JSON。

### 接入已有 pytest 文件

保留原有 pytest 写法和精度断言，将清单条目的 `runner` 设为 `pytest` 即可，无需实现上面的 benchmark 接口：

```json
{
  "name": "zhangsan_add",
  "runner": "pytest",
  "driver": "zhangsan/test_add.py",
  "arguments": []
}
```

pytest 适配器保留原断言，并验收 Auto 耗时预测。用例需实际运行 kernel 并检查精度；直接调用编译后 kernel 等特殊形式可能需要专门适配。`UNSUPPORTED` 不算通过。

## 5. 运行与查看结果

以下命令在仓库根目录执行。先完成个人清单登记，再按需选择：

### 查看清单与执行计划

```bash
# 只查看，不运行 NPU 测试
python costmodel_tests/run.py list --category personal
python costmodel_tests/run.py --category personal \
  --case custom_my_add_f32_n1024 --device 0 --dry-run
```

### 运行一个用例

```bash
python costmodel_tests/run.py --category personal \
  --case custom_my_add_f32_n1024 --device 0 --retries 3
```

运行上面的 pytest 示例时，将 case 名换为 `zhangsan_add`。

### 运行个人清单中的全部用例

```bash
# 单卡串行
python costmodel_tests/run.py --category personal --device 0

# 可选：单卡并发，失败后额外重试 3 次
python costmodel_tests/run.py --category personal --device 0 --jobs 2 --retries 3
```

`--retries` 默认 `3`，只重试失败用例；设为 `0` 可关闭重试。并发可能影响实测耗时，严格验收耗时误差时使用串行命令。

### 查看结果

打开终端打印的结果目录，优先阅读 `report.md`：

| 检查项 | 当前运行器的判定 |
|---|---|
| 计算精度 | 满足参考结果和用例自身的精度容差 |
| 耗时预测 | 默认检查 Auto 所选路线，预测耗时误差的绝对值 ≤ 10% |

benchmark 可用 `--validation-scope all` 增加显式路线检查；pytest 仍验收 Auto。当前不要求 Auto 是实测最快路线。

完整参数、结果文件和排查方法见 [测试使用说明](README.md)。
