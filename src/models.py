"""FunASR 模型封装。"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from funasr import AutoModel


DEFAULT_MODELSCOPE_CACHE = Path.home() / ".cache/modelscope/hub/models/iic"
DEFAULT_MODEL_CACHE_NAMES = {
    "fsmn-vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "paraformer-zh": (
        "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-"
        "vocab8404-pytorch"
    ),
    "ct-punc": "punc_ct-transformer_cn-en-common-vocab471067-large",
    "fa-zh": "speech_timestamp_prediction-v1-16k-offline",
}


class FunASRModels:
    """统一提供 VAD、ASR、标点恢复和强制对齐能力。

    每次推理只加载当前需要的模型，结束后立即释放，避免四个模型同时
    占用内存或显存。
    """

    def __init__(
        self,
        device: str | None = None,
        cache_dir: str | Path = DEFAULT_MODELSCOPE_CACHE,
        model_cache_names: dict[str, str] | None = None,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir).expanduser()
        self.model_cache_names = (
            model_cache_names.copy()
            if model_cache_names is not None
            else DEFAULT_MODEL_CACHE_NAMES.copy()
        )

    def is_ready(self) -> bool:
        """检查四个模型的本地缓存文件是否完整。"""
        for cache_name in self.model_cache_names.values():
            model_dir = self.cache_dir / cache_name
            if not (model_dir / "config.yaml").is_file():
                return False
            if not (model_dir / "model.pt").is_file():
                return False
        return True

    def resolve_model(self, name: str) -> str:
        """优先使用完整的本地缓存，否则返回可联网下载的模型别名。"""
        cache_name = self.model_cache_names.get(name)
        if cache_name is not None:
            cache_path = self.cache_dir / cache_name
            if (cache_path / "config.yaml").is_file() and (
                cache_path / "model.pt"
            ).is_file():
                print(f"使用本地缓存: {cache_path}")
                return str(cache_path)

        print(f"本地没有完整缓存，将通过 ModelScope 下载: {name}")
        return name

    def _load(self, name: str) -> AutoModel:
        print(f"\n加载模型: {name}（设备: {self.device}）")

        # 只调整 VAD 的尾部静音判停时间。连续检测到约 700ms 静音时
        # 结束当前语音段；不设置最长片段时间，避免按固定时长硬切。
        model_kwargs: dict[str, Any] = {}
        if name == "fsmn-vad":
            model_kwargs["max_end_silence_time"] = 700

        return AutoModel(
            model=self.resolve_model(name),
            device=self.device,
            disable_update=True,
            **model_kwargs,
        )

    @staticmethod
    def _clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, model_name: str, **kwargs: Any) -> list[dict[str, Any]]:
        model = self._load(model_name)
        try:
            return model.generate(**kwargs)
        finally:
            del model
            self._clear_memory()

    def vad(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[dict[str, Any]]:
        """使用内存中的单声道波形检测语音时间区间。"""
        return self._generate(
            "fsmn-vad",
            input=audio_data,
            sample_rate=sample_rate,
        )

    def asr(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[dict[str, Any]]:
        """使用内存中的单声道波形识别文字。"""
        return self._generate(
            "paraformer-zh",
            input=audio_data,
            sample_rate=sample_rate,
        )

    def punctuate(self, text: str) -> list[dict[str, Any]]:
        """为文字恢复标点。"""
        return self._generate("ct-punc", input=text)

    def align(
        self,
        audio_data: np.ndarray,
        text: str,
        sample_rate: int = 16000,
    ) -> list[dict[str, Any]]:
        """使用内存波形将音频和文本对齐，返回字级时间戳。"""
        return self._generate(
            "fa-zh",
            input=(audio_data, text),
            data_type=("sound", "text"),
            sample_rate=sample_rate,
        )
    
