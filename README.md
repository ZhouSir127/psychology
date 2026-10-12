# MindTrack 大创项目系统落地学习原型

这是主线三的基础版本，围绕“提交采集数据 → 特征处理 → 模拟预测 → 保存结果 → 纵向展示”实现一个可运行闭环。适合已有编程和FastAPI基础、希望学习模型系统集成的同学。

**主线一的脑电、语音、人脸处理会实际读取文件和提取特征；主线二采用可控的模拟评分。** 默认文本采用四项统计占位，可切换为之前讨论的本地MiniLM模型。模拟分数不从特征计算，不具有心理风险含义。

## 最快运行

建议使用 Python 3.11 或 3.12（64位）。以下命令在本项目根目录执行。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_resources.py
.\.venv\Scripts\python.exe run.py
```

如果已安装的Python命令是 `python`，可用 `python -m venv .venv` 创建环境。无需激活虚拟环境，避免PowerShell执行策略问题。

Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_resources.py
.venv/bin/python run.py
```

打开 [工作台](http://127.0.0.1:8501)，选择“公开样例组合”，点击“提交处理”。首次使用librosa可能编译部分计算函数，处理时间会长于后续任务。进度每两秒刷新；完成后到“历史趋势”点击刷新。

接口文档：[FastAPI Swagger](http://127.0.0.1:8000/docs)。Ctrl+C停止服务。日志在 `runtime/backend.log` 和 `runtime/frontend.log`。

资源文件如果已随项目提供，下载脚本会跳过已有文件。安装依赖和首次下载需要联网；资源准备好后，基础流程不调用任何在线推理API。

如果PyPI下载很慢，可以在安装命令末尾加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 使用清华大学镜像。无需修改电脑的全局pip配置。

项目还提供 `requirements-tested.txt`，记录本次在Windows / Python 3.12下通过验证的确切依赖版本（包含测试依赖）。若希望复现本次环境，可将安装命令中的 `requirements.txt` 换成它。其他操作系统优先使用上面的基础依赖文件。验证范围见 [docs/验证记录.md](docs/验证记录.md)。

## 当前实现

- 三种二进制文件：EDF脑电、PNG/JPG人脸、WAV/OGG/FLAC语音；文本随元数据提交。
- 可只提交部分模态；记录缺失模态，不虚构补齐数据。
- 上传后返回任务编号，后台单线程串行处理，前端查询状态。
- 任务状态：queued → running → succeeded / failed；失败原因可查看并重试。
- SQLite保存采集时间、任务、结果和完整特征摘要；原文件独立保存在磁盘。
- 相同 subject_id + collection_id 返回原任务，避免采集端重传产生重复记录。
- 按人、日期查看记录和趋势，触发高等级提示，导出带模拟标记的CSV。
- `config.toml` 配置特征模块、预测模块、模型路径和分级阈值。

## 你应该如何学习

1. 先在页面提交三次样例，分别选择低、中、高等级，修改采集日期，观察趋势。
2. 打开 `app/contracts.py`，看一次采集、特征和预测结果的数据结构。
3. 看 `app/main.py` 的上传接口，再看 `app/db.py` 如何写入SQLite。
4. 看 `app/pipeline.py`：这里只负责调用模块和更新状态，不包含算法。
5. 看两个 `app/adapters/` 模块，理解未来队友实现的替换位置。
6. 看 `dashboard.py` 怎样通过HTTP查询，最后阅读 `tests/test_api.py`。

```text
Streamlit / 模拟采集客户端
           │ HTTP上传或提交公开样例
           ▼
FastAPI → SQLite任务表 → 单个后台工作线程
                              │
                         FeatureExtractor  ← 主线一
                              │ FeatureBundle
                           Predictor       ← 主线二
                              │ Prediction
                      SQLite结果与特征
                              │ HTTP查询
                    页面趋势 / 提示 / CSV
```

## 可选：真实中文语义向量

默认的simple模式无需下载约500MB文本权重；可以先完成系统学习，再开启MiniLM。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-text.txt
.\.venv\Scripts\python.exe scripts/download_resources.py --text
```

在 `config.toml` 把 `text_mode = "simple"` 改为 `text_mode = "minilm"`，停止后重新运行 `run.py`。模型从本地路径加载并在CPU上执行，不会在服务处理时自动下载。它输出语义向量，不是情绪或心理风险分数。

## 模拟采集端自动上传

保持后端运行，在另一个终端执行：

```powershell
.\.venv\Scripts\python.exe scripts/replay_client.py --subject demo_01 --count 5 --interval 3
```

该客户端每3秒上传一次真实EDF样例与文本，轮换五种模拟分数。未来把文件读取换成主线一输出的切片，即可沿用HTTP提交结构。上传频率和推理耗时独立；队列满时后端返回429。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

单元/接口测试使用独立临时数据库和轻量特征模块，不依赖下载资源，也不更改自己的历史记录。覆盖真实任务调度、去重、失败重试、时间筛选、导出、重启恢复、文件校验与模块替换。

要验证下载的真实四模态样例，在PowerShell中设置 `$env:MINDTRACK_TEST_RESOURCES = "1"` 后再次运行pytest；Linux使用 `MINDTRACK_TEST_RESOURCES=1 .venv/bin/python -m pytest -q`。完整流程测试也使用临时数据库。

## 本版边界

- 只监听127.0.0.1，供单机学习使用。匿名编号是记录归属，不是登录认证；添加权限控制前不要开放给外部用户。
- 仅运行一个后端进程。不要加 `--workers`；多进程会让内存任务队列和启动恢复策略互相干扰。
- 使用工作线程隔离请求处理与计算。没有持久化消息队列；意外重启会把未完成任务标为失败，由页面显式重试。
- 支持已经截断的文件，不包括设备SDK、LSL同步、实时视频和完整伪迹清理。主线一负责这些内容。
- 公开样例不是同一个人的同步四模态数据。它们仅用于通路验证；不会用于科研训练或有效性评估。
- 人脸blendshape不等于FACS AU；脑电示例是64通道而不是项目设备的128通道；代码读取真实通道数。
- 预测历史入口目前提供更早的分数/特征摘要。跨会话状态、原始长序列和个体模型由主线二与团队进一步约定。
- 导出/查询最多10000条，模型历史最多1000条；适合本项目小规模演示，扩大规模时再加分页与存储策略。

完整接口及替换说明见 [docs/接口与替换.md](docs/接口与替换.md)。资源来源见 [resources/SOURCES.md](resources/SOURCES.md)。
