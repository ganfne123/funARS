"""FunASR 各模型的推理冒烟测试。

运行方式：
    uv run python -m unittest tests/test_models.py -v

首次运行时，如果本地没有完整缓存，FunASR 会自动下载模型。
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from src.models import FunASRModels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_PATH = PROJECT_ROOT / "ttsmaker-file-2026-8-26-0-9-55.mp3"


@unittest.skipUnless(
    os.getenv("RUN_FUNASR_INTEGRATION") == "1",
    "真实 FunASR 冒烟测试默认跳过；设置 RUN_FUNASR_INTEGRATION=1 后运行。",
)
class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not AUDIO_PATH.is_file():
            raise unittest.SkipTest(f"音频文件不存在: {AUDIO_PATH}")

        # 视频/MP3 中提取的音轨可能是 44.1 kHz 或 48 kHz，而 FunASR 的
        # VAD、ASR 和 FA-ZH 模型统一使用 16 kHz 单声道音频。测试开始前
        # 先完成解码、单声道转换和真实重采样，不能只修改采样率变量。
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.audio_path = Path(cls.temp_dir.name) / "audio16k.wav"
        audio_data, original_sample_rate = sf.read(
            AUDIO_PATH,
            dtype="float32",
        )

        # soundfile 的多声道形状是 (采样点数, 声道数)，对声道维求平均。
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        sample_rate = 16000
        if original_sample_rate != sample_rate:
            audio_tensor = torch.from_numpy(audio_data)
            audio_data = AF.resample(
                audio_tensor,
                orig_freq=original_sample_rate,
                new_freq=sample_rate,
            ).numpy()

        cls.audio_data = np.asarray(audio_data, dtype=np.float32)
        cls.sample_rate = sample_rate
        sf.write(cls.audio_path, audio_data, sample_rate)

        print(f"原始测试音频: {AUDIO_PATH}")
        print(f"原始采样率: {original_sample_rate} Hz")
        print(f"模型测试音频: {cls.audio_path}")
        print(f"模型采样率: {sample_rate} Hz")
        print(f"音频时长: {len(audio_data) / sample_rate:.2f} 秒")

        if sample_rate != 16000:
            raise RuntimeError(f"测试音频重采样失败: {sample_rate} Hz")

        cls.models = FunASRModels()

    @classmethod
    def tearDownClass(cls) -> None:
        # 四个模型测试结束后删除临时 16 kHz WAV 文件。
        cls.temp_dir.cleanup()

    def test_vad(self) -> None:
        """VAD 应返回至少一个语音时间区间。"""
        result = self.models.vad(
            self.audio_data,
            sample_rate=self.sample_rate,
        )
        print("VAD 结果:", result)
        self.assertTrue(result and "value" in result[0])

    def test_asr(self) -> None:
        """ASR 应从测试音频中识别出文字。"""
        result = self.models.asr(
            self.audio_data,
            sample_rate=self.sample_rate,
        )
        print("ASR 结果:", result)
        self.assertTrue(result and result[0].get("text"))
        plain_text=result[0]["text"].replace(" ", "")
        punctuationText=self.models.punctuate(plain_text)
        print("标点恢复结果:", punctuationText)

    def test_alignment(self) -> None:
        """强制对齐模型应根据 ASR 文本返回字级时间戳。"""
        asr_result = self.models.asr(
            self.audio_data,
            sample_rate=self.sample_rate,
        )
        self.assertTrue(asr_result and asr_result[0].get("text"))

        result = self.models.align(
            self.audio_data,
            asr_result[0]["text"],
            sample_rate=self.sample_rate,
        )
        print("强制对齐结果:", result)
        self.assertTrue(result and result[0].get("timestamp"))


if __name__ == "__main__":
    unittest.main()
