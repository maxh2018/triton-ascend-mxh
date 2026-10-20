# CostModel 测试资料仓

本分支仅保存 CostModel 测试用例、测试依赖、profile 和说明文档。
它不再包含可构建的完整 Triton-Ascend 编译器；请使用单独安装的兼容编译环境运行测试。

- [测试脚本使用说明](costmodel_tests/README.md)
- [自定义测试用例提交规范](costmodel_tests/CUSTOM_CASES.md)

无需 NPU 的资料自检：

```bash
python -m unittest discover -s costmodel_tests -p 'test_*.py' -v
```
