# CostModel 测试资料仓

本分支仅保存 CostModel 测试用例、测试依赖、profile 和说明文档。
它不再包含可构建的完整 Triton-Ascend 编译器；请使用单独安装的兼容编译环境运行测试。

- [测试脚本使用说明](costmodel_tests/README.md)
- [用例来源记录](costmodel_tests/sources.json)
- [必须保留的本地测试依赖](costmodel_tests/dependencies.json)

包含 pytest_ut 的 374 个测试文件，以及 PR #2528 的 100-case 回归矩阵。
加载 NPU 编译环境后，在仓库根目录执行：

```bash
python costmodel_tests/run.py list
python costmodel_tests/run.py case --case softmax_original --device 0
python costmodel_tests/run.py matrix --device 0
python costmodel_tests/run.py pytest --device 0
```

`backend/`、`python/triton/runtime/` 和 `LegacyMemoryAccess/` 下保留的少量文件是
源码合同测试直接读取的 fixture，不构成编译器安装包。原编译器完整源码仍可从 main 分支取得。

无需 NPU 的资料自检：

```bash
python -m unittest discover -s costmodel_tests -p test_offline.py -v
```
