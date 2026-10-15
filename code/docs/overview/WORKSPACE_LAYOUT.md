# 公开仓库布局

从仓库根目录执行 `python -m pip install -e ./code` 安装案例库。
Python、pytest 及研究管线命令通常在 `code/` 中运行；带路径参数的脚本可显式指定输入和输出。

| 目录 | 用途 |
|---|---|
| `doe_testbed/` | 案例实现，按6个领域分类；证据说明与源码相邻 |
| `benchmark/` | 方法、指标、分析和正式数据；`results/`存正式运行，`data/`存Sobol等资料 |
| `scripts/` | 分析、生成和检查程序；按其README中的用途查找 |
| `tests/`、`benchmark/tests/` | 物理和方法/管线测试 |
| `examples/`、`tools/` | 使用示例、运行维护工具 |
| `docs/overview/` | 架构和项目概览 |
| `docs/project/` | 更新记录、贡献、行为及安全说明 |
| `docs/` | 当前复算、证据、统计政策、数据字典 |
| `metadata/` | 软件和数据登记元数据；其存在不代表已取得存档 DOI |

上述相对路径均以 `code/` 为基准。仓库根目录的 `manuscripts/` 保存图表源数据、
参考图件、补充表格、冻结协议和数值汇总；`project_docs/` 保存已完成分析的派生表格。

图表源数据、绘图、数值汇总和补充表格脚本的默认路径统一配置在
`scripts/project_layout.py`，指向仓库根目录的 `manuscripts/`，不依赖调用时的工作目录；
显式路径参数仍可覆盖默认值。完整管线使用相同目录，smoke 管线的输出则留在自己的运行目录中。
具体数据入口和复现边界见[根目录说明](../../../README.md)。
