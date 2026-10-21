# 算珍仓用例

本目录预留给后续导入的算珍仓算子。建议按算子建子目录，并在 `manifest.json` 中登记。
`driver` 路径相对于本清单；与 CI、个人用例分开保存。

```text
cases/suanzhen/
  manifest.json
  <算子名>/
    bench_<算子名>.py
```

```bash
python costmodel_tests/run.py list --category suanzhen
python costmodel_tests/run.py matrix --category suanzhen --device 0
```

当前清单为空，登记用例后才能运行。每个脚本头部注明原仓库、原文件路径和版本。
接口要求见 [提交规范](../../costmodel_tests/CUSTOM_CASES.md)。
