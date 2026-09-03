from fastapi import HTTPException
import asyncio
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from typing import List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
import json

from src.models import FunASRModels
from src.Entities import *


def process_audio_task(
        task_id: str,
        audio_path: Path,
        save_segments: bool = False,
) -> None:
    task_info = tasks.get(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if audio_path is None:
        raise HTTPException(status_code=400, detail="音频文件路径无效")

        