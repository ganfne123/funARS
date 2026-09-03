# funARS

基于 [FunASR](https://github.com/modelscope/FunASR) 和 FastAPI 的异步语音转写服务。上传音频后，服务会在后台完成语音活动检测（VAD）、语音识别、标点恢复和时间戳生成，并提供任务查询与结果下载接口。

> 这是一个面向中文音频处理的开发中项目。服务入口当前固定使用 `cuda:0`，请在具备可用 NVIDIA GPU 与对应 PyTorch/CUDA 环境的机器上运行。

## 功能

- 异步上传音频并创建后台处理任务
- 按 VAD 语音区间进行中文 ASR 与标点恢复
- 将音频统一转换为单声道、16 kHz 后推理
- 长语音段使用 FA-ZH 强制对齐，生成更细粒度的句子时间范围
- 查询任务状态和转写结果，下载时间戳 JSON 文件
- 模型按推理阶段加载、释放，降低多个模型同时占用显存的压力

## 处理流程

```text
上传音频
  │
  ├─► 单声道转换与 16 kHz 重采样
  ├─► FSMN-VAD 语音分段
  ├─► Paraformer 中文识别
  ├─► CT-Transformer 标点恢复
  ├─► 长片段：FA-ZH 强制对齐
  └─► 任务结果与 timestamps JSON
```

## 环境要求

- Python 3.11 或更高版本（项目使用 `.python-version` 固定为 3.11）
- [uv](https://docs.astral.sh/uv/)
- 可用的 NVIDIA GPU、CUDA 驱动和与 PyTorch 兼容的运行环境
- 首次运行时可访问 ModelScope；模型会自动下载到 `~/.cache/modelscope/hub/models/iic`

项目依赖中包含 PyTorch、FunASR 和音频处理库，首次安装与模型下载可能需要较长时间和较大的磁盘空间。

## 快速开始

```bash
git clone git@github.com:ganfne123/funARS.git
cd funARS
uv sync
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

开发时启用自动重载：

```bash
uv run uvicorn src.app:app --reload
```

服务启动后可在 <http://127.0.0.1:8000/docs> 查看 OpenAPI 交互文档。

## API

### 创建转写任务

```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/tasks?save_segments=false' \
  -F 'file=@./example.wav'
```

响应示例：

```json
{
  "task_id": "b14b0d98-425a-4fb9-8e66-f2818b660742",
  "status": "pending"
}
```

`save_segments` 已作为接口参数保留在任务结果中；当前实现不会单独写出音频片段。

### 查询任务

```bash
curl 'http://127.0.0.1:8000/api/v1/tasks/<task_id>'
```

任务状态为 `pending`、`processing`、`completed` 或 `failed`。完成后，响应中的 `result.segments` 包含句子文本以及 `start_ms`、`end_ms`（毫秒）时间范围。

### 下载时间戳文件

```bash
curl -OJ 'http://127.0.0.1:8000/api/v1/download/<task_id>/timestamps'
```

仅已完成的任务可以下载。文件保存在服务端 `src/outputs/<task_id>/<task_id>.timestamps.json`，结构如下：

```json
{
  "all_sentence_timestamps": [
    ["你好", [0, 820]],
    ["世界", [820, 1560]]
  ]
}
```

## 使用的模型

| 阶段 | FunASR 模型 |
| --- | --- |
| 语音活动检测 | `fsmn-vad` |
| 语音识别 | `paraformer-zh` |
| 标点恢复 | `ct-punc` |
| 强制对齐 | `fa-zh` |

模型缓存完整时会优先从本地加载；缓存不完整时，FunASR 会通过 ModelScope 下载对应模型。

## 项目结构

```text
src/
  app.py          # FastAPI 路由
  main.py         # 后台任务与转写主流程
  models.py       # FunASR 模型封装
  workflow.py     # 音频、文本与时间戳工具
  Entities.py     # 任务状态与数据模型
tests/
  test_project.py # API、模型封装与工作流回归测试
  test_models.py  # 真实模型冒烟测试（默认跳过）
  test_workflow.py# 真实工作流测试（默认跳过）
```

## 测试

运行不加载真实模型的项目测试：

```bash
UV_CACHE_DIR=/tmp/funasr-api-uv-cache \
  uv run python -m unittest discover -s tests -p 'test_project.py' -v
```

真实模型测试默认跳过；需要下载模型、使用 GPU，并显式设置环境变量：

```bash
RUN_FUNASR_INTEGRATION=1 \
  uv run python -m unittest tests/test_models.py -v
```

## 当前限制

- 任务状态保存在进程内存中，服务重启后无法恢复历史任务。
- 上传文件会在任务结束时删除；生成的时间戳文件保存在本地磁盘。
- 当前项目测试中仍有部分用例引用旧的 `src.main.workflow` 接口，需要在后续重构中同步更新。
