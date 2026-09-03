"""后台任务共享状态。"""
from pathlib import Path

from datetime import datetime
from enum import Enum
from typing import Any,List,Dict,Tuple
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = Path(ROOT_DIR / "uploads")
OUTPUT_DIR = Path(ROOT_DIR / "outputs")

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    filename: str
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None



tasks: dict[str, TaskInfo] = {}


TimestampMs = int  # 以毫秒为单位的时间戳
AudioSegment = np.ndarray  # 音频采样数据
VadSegment = Tuple[TimestampMs, TimestampMs]  # VAD片段， 前为start_ms，后为end_ms
WordTimestamp = Tuple[str, List[TimestampMs]]  # 词时间戳
SentenceTimestamp = Tuple[str, List[TimestampMs]]  # 句子时间戳


class CreateTaskResponse(BaseModel):
    """创建任务响应"""
    task_id: str
    status: TaskStatus

class TaskResponse(BaseModel):
    """创建任务"""
    task_id: str
    upload_path: Path


@dataclass
class ProcessingResult:
    """处理结果数据结构"""
    segments: List[SentenceTimestamp]
    vad_segments: List[VadSegment]
    char_timestamps: List[WordTimestamp]
    metadata: Dict[str, Any]

OUTPUT_DIR = Path(__file__).parent.resolve() / "outputs"

SENTENCE_DELIMITERS: Tuple[str, ...] = (
    "。",
    "！",
    "？",
    "；",
    "，",
    "、",
    "：",
    "“",
    "”",
    "‘",
    "’",
    '"',
    "'",
)


