# 自定义测试用例提交规范

**只需提交两部分：一个能独立运行的 Python 脚本，以及清单中的一条配置。**

## 现成的最小例子：向量加法

- [示例脚本：x + y](../third_party/ascend/unittest/costmodel_ut/costmodel_generalization/bench_custom_add_example.py)
- [示例清单：1 个 case](../third_party/ascend/unittest/costmodel_ut/costmodel_generalization/custom_manifest.example.json)

示例 JSON 中每个字段的含义：

| 字段 | 示例值 | 含义 |
|---|---|---|
| `cases` | `[{...}]` | 用例列表，每个对象代表一个测试配置；增加对象即可登记多个 case。 |
| `name` | `custom_add_f32_n1024` | 唯一的用例名，运行命令的 `--case` 就填写这个值。 |
| `family` | `elementwise` | 算子类别，用于分类和按类别统计；这里表示逐元素计算。 |
| `split` | `holdout` | 数据用途：`train` 表示用于模型开发，`holdout` 表示留出验证；新提交默认填 `holdout`，不改变运行方式。 |
| `driver` | `bench_custom_add_example.py` | 执行该 case 的 Python 脚本路径，相对于 `costmodel_generalization/` 目录。 |
| `arguments` | `["--n-elements", "1024"]` | 传给脚本的自定义命令行参数，按顺序填写，所有元素均为字符串；这里表示测试 1024 个元素。 |

例如，上述 `driver` 和 `arguments` 会组成
`python bench_custom_add_example.py --n-elements 1024`，运行器还会自动追加 route、factor、
report 等公共参数。相同脚本可以登记多个 case，例如修改 `name` 和 `--n-elements` 的值来测试不同输入规模。

加载兼容的 NPU 编译环境后，在仓库根目录直接运行：

```bash
python costmodel_tests/run.py case \
  --manifest third_party/ascend/unittest/costmodel_ut/costmodel_generalization/custom_manifest.example.json \
  --case custom_add_f32_n1024 --device 0
```

脚本依次完成：生成输入 → 运行 x + y → 检查精度 → 采集耗时 → 输出 JSON。
统一入口再负责五条路线、失败重试和 ±10% 耗时预测验收。
例子展示完整接口，不表示当前 profile 对这个新 shape 已经实测通过。
添加自己的算子时，复制这两个文件，主要替换 kernel、参考结果和清单里的名字/参数。

## 提交规则

建议复制现有
[bench_muls_add_costmodel.py](../third_party/ascend/unittest/costmodel_ut/costmodel_generalization/bench_muls_add_costmodel.py)，
作为提交模板，只需替换其中 kernel、输入和参考结果，保留参数解析、选路、正确性检查、计时和 JSON 输出部分。

## 1. 文件放哪里

以下路径均相对于 `third_party/ascend/unittest/costmodel_ut/costmodel_generalization/`：

```text
bench_custom_my_add.py      # 你的 kernel 和测试脚本
custom_manifest.json       # 自定义 case 清单；多个用例可以共用
```

脚本放在这个目录，便于直接复用 `profile_utils.py`。
用例名用英文字母、数字、下划线，且不重名，例如 `custom_my_add_f32_n1024`。
脚本头部用注释注明来源、额外依赖和适用芯片；输入尽量在脚本里生成。
如果需要辅助文件，一起提交并使用相对脚本位置的路径读取。
不要提交机器绝对路径、编译缓存、profiler 数据和运行日志。

## 2. 脚本遵守什么接口

脚本必须接受这些参数，含义与现有 benchmark 一致：

| 参数 | 要求 |
|---|---|
| `--route` | 支持 `auto`、`simd`、`simt_only` |
| `--factor` | 支持 `1`、`2`、`4`，传给 kernel 的 `superblock_factor` |
| `--warmup`、`--repeats` | 预热和重复计时次数 |
| `--report` | Auto 模式下编译器选路报告的输出路径 |
| `--profile-dir` | 可选；提供时采集 NPU profiler，不提供时也应能运行 |

自己的 shape、dtype 等参数可以额外添加，例如 `--n-elements 1024`。
一次 case 对应一个被测 kernel；同一 kernel 的不同 shape 可以登记为多个 case。

**必须做三件事：**

1. 按 `--route` 和 `--factor` 编译运行，不能在脚本内固定一种路线。
   Auto 使用 `compile_mode="simd_simt"`、`auto_simt_scope_mode="auto"`，
   并把 `--report` 传给 `auto_simt_scope_dump`；显式路线关闭自动选路。
   `logical_program_count_hint` 应填真实逻辑 program 数，物理核数从设备属性获取。
2. 用参考结果检查输出，例如 `torch.testing.assert_close()`；失败抛异常或返回非零退出码。
   不同路线保持相同计算语义，不允许跳过不支持的路线后伪报成功。
3. 提供 `--profile-dir` 时复用 `profile_kernel(launch, args.profile_dir, kernel_name)`
   采集实际耗时。最后向标准输出打印一个 JSON 对象，至少包含：

```python
print(json.dumps({
    "case": "custom_my_add_f32_n1024",
    "correct": True,        # 只有真实精度检查通过后才输出 True
    "profiler": profiler,   # profile_kernel() 的返回值；未采集时为 None
}))
```

`profiler` 的返回值包含实际测得的 `average_us`，单位为微秒。
允许前面打印日志，但最终 JSON 后不要再打印其他 JSON。
选路报告由兼容编译器生成，不能自行填写预测时间；没有报告就无法通过 CostModel 审计。

## 3. 清单怎么写

在 `custom_manifest.json` 中登记脚本和参数：

```json
{
  "cases": [
    {
      "name": "custom_my_add_f32_n1024",
      "family": "elementwise",
      "split": "holdout",
      "driver": "bench_custom_my_add.py",
      "arguments": ["--n-elements", "1024"]
    }
  ]
}
```

`name` 是运行时选择的 case 名；`family` 表示算子类别；新上传 case 的 `split` 默认填
`holdout`。`driver` 相对于上述 benchmark 目录；`arguments` 只填写自定义参数，
不要重复填写 route、factor、report、profile-dir、warmup、repeats。
暂不需要 Mixed 等额外模式，后续有需求再补充。
**自定义 case 使用单独清单，保留原 PR 的 100-case 清单不变。**

## 4. 怎么验证提交格式

加载兼容的 NPU 编译环境后，在仓库根目录执行：

```bash
# 路径较长，先定义一个变量
MANIFEST=third_party/ascend/unittest/costmodel_ut/costmodel_generalization/custom_manifest.json

# 查看清单及命令（不运行 NPU 测试）
python costmodel_tests/run.py list --manifest "$MANIFEST"
python costmodel_tests/run.py case --manifest "$MANIFEST" \
  --case custom_my_add_f32_n1024 --device 0 --dry-run

# 运行并验收一个自定义 case
python costmodel_tests/run.py case --manifest "$MANIFEST" \
  --case custom_my_add_f32_n1024 --device 0 --retries 3

# 批跑该清单；并发和重试沿用现有参数
python costmodel_tests/run.py matrix --manifest "$MANIFEST" \
  --devices 0,1 --jobs 2 --retries 3
```

系统会测试 Auto、SIMD、SIMT F1/F2/F4。计算结果正确且每条路线的预测耗时误差
不超过 ±10% 才通过；“Auto 是否实测最快”不属于当前审计条件。
如果只有原始 pytest、尚未接入上述接口，先放入 `pytest_ut/`，只能按 pytest 原有断言运行，
不会自动具备 CostModel 计时审计能力。
