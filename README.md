# CostModel 测试资料仓

本分支仅保存 CostModel 测试用例、测试依赖、profile 和说明文档。
它不再包含可构建的完整 Triton-Ascend 编译器；请使用单独安装的兼容编译环境运行测试。

- [测试脚本使用说明](costmodel_tests/README.md)
- [自定义测试用例提交规范](costmodel_tests/CUSTOM_CASES.md)

用例按来源分目录保存：

| 分类 | 路径 | 说明 |
|---|---|---|
| CI 用例 | [cases/ci/](cases/ci/README.md) | 现有 pytest_ut、PR 的 100-case 和测试依赖 |
| 个人用例 | [cases/personal/](cases/personal/README.md) | 个人算子上传到 `<姓名>/` 子目录，登记到此处 `manifest.json` |
| 算子仓用例 | [cases/operator_repos/](cases/operator_repos/README.md) | 按 vLLM、SGLang 等来源仓库分目录保存 |

**优先执行 `--manifest` 指定的清单，可跨目录组合用例。** 未指定清单时，`--category` 才用于查找分类的 `manifest.json`；两者都不指定时执行 `cases/ci/manifest.json`（原始 100 个 case）。
`cases/ci/manifest_all.json` 是尚未验证的 CI 全量清单，只有显式指定才运行。

```bash
python costmodel_tests/run.py --device 0
python costmodel_tests/run.py --manifest cases/ci/manifest_all.json --device 0
```

无需 NPU 的资料自检：

```bash
python -m unittest discover -s costmodel_tests -p 'test_*.py' -v
```
