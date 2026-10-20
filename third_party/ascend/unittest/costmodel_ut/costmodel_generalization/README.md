# CostModel 回归测试入口

本目录收录 PR #2528 固定提交的 100-case 矩阵、benchmark、运行和验收脚本。

完整说明：[CostModel 测试用例与脚本使用说明](../../../../../costmodel_tests/README.md)。

在仓库根目录执行：

```bash
python costmodel_tests/run.py list
python costmodel_tests/run.py case --case softmax_original --device 0
python costmodel_tests/run.py matrix --device 0
python costmodel_tests/run.py pytest --device 0
```

请先加载兼容 PR 的 NPU 编译环境。100-case 的通过条件包括计算精度和 ±10% 耗时预测误差，
没有直接断言 Auto 必须是实测最快路线。
