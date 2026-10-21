# CostModel 回归测试入口

本目录收录 PR #2528 固定提交的 100-case 矩阵、benchmark、运行和验收脚本。

完整说明：[CostModel 测试用例与脚本使用说明](../../../../../../../costmodel_tests/README.md)。

在仓库根目录执行：

```bash
python costmodel_tests/run.py list
python costmodel_tests/run.py case --case softmax_original --device 0
python costmodel_tests/run.py matrix --device 0
python costmodel_tests/run.py matrix --devices 0,1 --jobs 2 --retries 3
python costmodel_tests/run.py pytest --device 0
```

请先加载兼容 PR 的 NPU 编译环境。默认只检查 Auto 最终选择路线的计算精度和 ±10%
耗时预测误差，与原 PR 的 100-case 结果表范围一致。加 `--validation-scope all` 可扩展到
完整路线验收。没有直接断言 Auto 必须是实测最快路线。
