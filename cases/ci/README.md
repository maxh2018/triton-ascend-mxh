# CI 用例

现有 pytest_ut、PR #2528 的 100-case 和必要依赖统一保存在本目录。
内部保留原相对目录层级，避免破坏已有测试的源码读取和导入关系。

在仓库根目录运行：

```bash
python costmodel_tests/run.py list --category ci
python costmodel_tests/run.py matrix --category ci --device 0
python costmodel_tests/run.py pytest --category ci --device 0
```

`ci` 是默认分类，不加 `--category` 的命令仍运行 CI 用例。
个人上传的算子放到 `cases/personal/`，不要混入本目录。

[完整使用说明](../../costmodel_tests/README.md)
