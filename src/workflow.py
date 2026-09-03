from fastapi import UploadFile
import asyncio
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from typing import List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import uuid

from src.models import FunASRModels
from src.Entities import *

class audioProcessor:
    def __init__(self):
        self.sample_rate = 16000  # 全局采样率，单位 Hz


    def prepare_audio(
        self,
        audio_path: Path,
    ) -> Tuple[AudioSegment, int]:
        """读取音频并返回单声道波形数组和原始采样率。"""
        audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")

        # soundfile 的多声道形状为 (采样点数, 声道数)，对声道维求平均。
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1, dtype=np.float32)

        return audio_data, sample_rate


    def merge_model_text(
        self,
        results: List[Dict[str, Any]],
        *,
        remove_whitespace: bool = False,
    ) -> str:
        """合并 FunASR 模型返回的 text 字段。"""
        texts = (str(result.get("text", "")) for result in results)
        if remove_whitespace:
            return "".join("".join(text.split()) for text in texts)
        return "".join(texts).strip()

    def ms_to_samples(self, ms: TimestampMs) -> int:
        """毫秒转换为采样点, 采样点为音频索引"""
        return int(ms * self.sample_rate / 1000)

    def remove_punctuation(self, text: str) -> str:
        """移除用于分句的标点符号。"""
        translation = str.maketrans("", "", "".join(SENTENCE_DELIMITERS))
        return text.translate(translation)


    def serialize_sentences(self, text: str) -> List[str]:
        """按标点切分文本，并过滤空句子。"""
        translation = str.maketrans(
            {delimiter: "|" for delimiter in SENTENCE_DELIMITERS}
        )
        serialized_text = text.translate(translation)
        return [sentence.strip() for sentence in serialized_text.split("|") if sentence.strip()]


    def align_sentence_timestamps(
        self,
        word_timestamps: List[WordTimestamp],
        sentences: List[str],
    ) -> List[SentenceTimestamp]:
        """对齐句子时间戳

        将单词级别的时间戳对齐到句子级别
        """
        if len(word_timestamps) != len("".join(sentences)):
            raise ValueError(
                f"单词时间戳数量({len(word_timestamps)})与句子总字符数({len(''.join(sentences))})不匹配"
            )

        aligned_timestamps: List[SentenceTimestamp] = []
        char_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len == 0:
                continue

            # 获取句子首字符和末字符的时间戳
            start_time = word_timestamps[char_index][1][0]
            end_time = word_timestamps[char_index + sentence_len - 1][1][1]

            aligned_timestamps.append((sentence, [start_time, end_time]))
            char_index += sentence_len

        return aligned_timestamps





    def generate_uniform_char_timestamps(
            self,
            text: str, start_ms: TimestampMs, end_ms: TimestampMs
        ) -> List[WordTimestamp]:
            """为短片段生成平均分配的字符时间戳"""
            char_timestamps: List[WordTimestamp] = []
            text_len = len(text)

            if text_len == 0:
                return char_timestamps

            duration_per_char = (end_ms - start_ms) / text_len

            for i, char in enumerate(text):
                char_start = int(start_ms + i * duration_per_char)
                char_end = int(start_ms + (i + 1) * duration_per_char)
                char_timestamps.append((char, [char_start, char_end]))

            return char_timestamps

    def fix_word_timestamps(
            self,
            word_timestamps: List[WordTimestamp],
            vad_segment: VadSegment,
        ) -> List[WordTimestamp]:
            """修正FA-ZH返回的单词时间戳"""
            if not word_timestamps:
                return []

            vad_start, vad_end = vad_segment

            # 计算首尾时间差异的平均调整量
            first_word_start_diff = word_timestamps[0][1][0] - vad_start
            last_word_end_diff = word_timestamps[-1][1][1] - vad_end
            avg_adjustment = (
                first_word_start_diff + last_word_end_diff
            ) / 2 + 10  # 微调漂移

            fixed_timestamps: List[WordTimestamp] = []
            for i, (word, timestamp) in enumerate(word_timestamps):
                start_time, end_time = timestamp

                if i == 0:
                    # 首个单词使用VAD起始时间
                    fixed_timestamps.append(
                        (word, [vad_start, int(end_time - avg_adjustment)])
                    )
                elif i == len(word_timestamps) - 1:
                    # 末尾单词使用VAD结束时间
                    fixed_timestamps.append(
                        (word, [int(start_time - avg_adjustment), vad_end])
                    )
                else:
                    # 中间单词调整时间戳
                    fixed_timestamps.append(
                        (
                            word,
                            [
                                int(start_time - avg_adjustment),
                                int(end_time - avg_adjustment),
                            ],
                        )
                    )

            return fixed_timestamps
    @staticmethod
    def save_audio(audio_data: AudioSegment, output_path: Path) -> None:
        """按指定路径保存音频"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, samplerate=16000)

    

    def write_timestamps_file(
            self,
        all_sentence_timestamps: list[tuple[str, list[int]]],
        output_path: Path,
    ) -> None:
        """将时间戳写入 JSON 文件"""
        payload = {
            "all_sentence_timestamps": all_sentence_timestamps,
        }

        temp_path = output_path.with_suffix(".json.part")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(output_path)

    


