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
| 算珍仓用例 | [cases/suanzhen/](cases/suanzhen/README.md) | 预留后续算珍仓导入用例 |

默认仍运行 CI 用例；运行个人用例使用 `--category personal`，算珍仓使用 `--category suanzhen`。

无需 NPU 的资料自检：

```bash
python -m unittest discover -s costmodel_tests -p 'test_*.py' -v
```
