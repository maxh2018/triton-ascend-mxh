# CI 用例

| 清单 | 内容 | 默认运行 |
|---|---|---|
| [manifest.json](manifest.json) | 原始 PR 100 个 case | 是 |
| [manifest_all.json](manifest_all.json) | 100 个配置 + CI 目录下 375 个测试文件 | 否，尚未验证 |

在仓库根目录执行：

```bash
# 默认：只运行原始 100 个 case
python costmodel_tests/run.py --device 0

# 查看全量清单；运行时去掉 list
python costmodel_tests/run.py list --manifest cases/ci/manifest_all.json
```

完整目录层级保留用于测试依赖。原始 PR 清单仍保存在深层目录，根目录 `manifest.json` 是路径已调整的副本。
全量清单通过显式 `--manifest` 选择，不会自动启用。更新清单：`python costmodel_tests/update_ci_manifests.py`。
个人算子放到 `cases/personal/<姓名>/`。

[使用说明](../../costmodel_tests/README.md)
