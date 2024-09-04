
# 股票（虚拟货币）投资因子挖掘平台

**作者**: ksjkz  
国内大学生,非科班,因热爱研究量化投资,希望通过量化把主观亏的赚回来

---

## 项目结构

### 主要目录

- **base**: 主要的库文件，定义了类和函数等
- **config**: 存放数据源的账号、URL 等信息（已在 `.gitignore` 中忽略个人信息）
- **feasible_ipynb**: 执行文件，相当于 `main`（目前尚未上传）

### base 目录下主要文件解析

- **agent_improve.py**: 利用 LLM 通过程序批量对话生成，迭代因子。
  
- **binance_trade_client.py**: 币安交易接口封装，提供数据和交易功能。
  
- **factor_generate.py**: 利用 LLM 生成无序因子，作为 `agent_improve` 的输入。
  
- **formulate_coding_AST.py**: 在 DataFrame 上进行因子计算，输入公式（字符串），由公式树解析并自动执行计算因子值。这个模块相对成熟。
  
- **formulate_filter_AST.py**: 利用公式树筛选复杂公式中的有效结构（该模块仍在起步阶段）。
  
- **get_tushare_data.py**: Tushare 数据接口封装，方便数据获取。
  
- **mapping_columns_name.py**: 公式映射，解决不同数据源 DataFrame 列名不一致的问题。
  
- **memory.py**: 一些零碎的常用操作（该文件会逐步更新）。
  
- **my_backtesting.py**: 单因子回测框架，方便测试因子表现。
  
- **my_df_func.py**: DataFrame 常用操作封装，提供便捷的功能。
  
- **my_os_func.py**: 操作系统常用操作封装，简化文件和目录操作。

---
目前最好模型: A股日频rankic 0.106  icir 37.90(近三年数据)
个人邮箱 qint20@126.com



