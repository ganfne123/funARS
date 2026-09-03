from __future__ import annotations

import numpy as np

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from dataclasses import dataclass

import soundfile as sf
from funasr import AutoModel

TimestampMs = int  # 以毫秒为单位的时间戳
AudioSegment = np.ndarray  # 音频采样数据
VadSegment = Tuple[TimestampMs, TimestampMs]  # VAD片段， 前为start_ms，后为end_ms
WordTimestamp = Tuple[str, List[TimestampMs]]  # 词时间戳
SentenceTimestamp = Tuple[str, List[TimestampMs]]  # 句子时间戳


@dataclass
class ModelConfig:
    asr_model_name: str = "paraformer-zh"
    vad_model_name: str = "fsmn-vad"
    ct_punc_model_name: str = "ct-punc"
    fazh_model_name: str = "fa-zh"
    device: str = "cuda:0"


@dataclass
class ProcessingResult:
    segments: List[SentenceTimestamp]
    vad_segments: List[VadSegment]
    char_timestamps: List[WordTimestamp]
    metadata: Dict[str, Any]


class AudioProcessor:
    """音频处理类"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def ms_to_samples(self, ms: TimestampMs) -> int:
        """毫秒转换为采样点, 采样点为音频索引"""
        return int(ms * self.sample_rate / 1000)

    def samples_to_ms(self, samples: int) -> TimestampMs:
        """采样点转换为毫秒"""
        return int(samples * 1000 / self.sample_rate)

    def extract_segment(
        self, audio_data: AudioSegment, start_ms: TimestampMs, end_ms: TimestampMs
    ) -> AudioSegment:
        """根据采样点提取音频片段"""
        start_sample = self.ms_to_samples(start_ms)
        end_sample = self.ms_to_samples(end_ms)
        return audio_data[start_sample:end_sample]

    @staticmethod
    def load_audio(audio_path: Path) -> Tuple[AudioSegment, int]:
        """加载音频文件，返回音频采样数据和采样率"""
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        audio_data, sample_rate = sf.read(str(audio_path))
        # 如果是立体声，转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data, sample_rate

    @staticmethod
    def save_audio(audio_data: AudioSegment, output_path: Path) -> None:
        """按指定路径保存音频"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, samplerate=16000)

    @staticmethod
    def resample_audio(
        audio_data: AudioSegment, original_rate: int, target_rate: int
    ) -> AudioSegment:
        """重采样音频采样数据到目标采样率"""
        import librosa

        return librosa.resample(
            audio_data, orig_sr=original_rate, target_sr=target_rate
        )


class ModelManager:
    """模型管理器"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.asr_model: Optional[AutoModel] = None
        self.vad_model: Optional[AutoModel] = None
        self.punc_model: Optional[AutoModel] = None
        self.fazh_model: Optional[AutoModel] = None
        self._loaded = False  # 模型加载 flag

    def load_models(self) -> None:
        if self._loaded:
            return

        self.asr_model = AutoModel(
            model=self.config.asr_model_name, device=self.config.device
        )
        self.vad_model = AutoModel(
            model=self.config.vad_model_name, device=self.config.device
        )
        self.punc_model = AutoModel(
            model=self.config.ct_punc_model_name, device=self.config.device
        )
        self.fazh_model = AutoModel(
            model=self.config.fazh_model_name, device=self.config.device
        )
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_models()


class SRTGenerator:
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        sentence_delimiters: Optional[List[str]] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.model_manager = ModelManager(self.model_config)
        self.audio_processor = AudioProcessor()
        self.sentence_delimiters = (
            sentence_delimiters
            if sentence_delimiters is not None
            else [
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
            ]
        )

    def load_models(self) -> None:
        self.model_manager.load_models()

    def get_vad_segments(
        self, audio_data: AudioSegment, sample_rate: Optional[int] = None
    ) -> List[VadSegment]:
        """获取VAD分段结果"""
        self.model_manager.ensure_loaded()
        sample_rate = sample_rate or self.audio_processor.sample_rate
        assert self.model_manager.vad_model is not None
        vad_result = self.model_manager.vad_model.generate(
            input=audio_data, sample_rate=sample_rate
        )
        if not vad_result or "value" not in vad_result[0]:
            return []
        return vad_result[0]["value"]

    def get_asr_text(self, audio_segment: AudioSegment) -> str:
        """获取ASR识别文本"""
        text = ""
        vad_segments = self.get_vad_segments(
            audio_segment, sample_rate=self.audio_processor.sample_rate
        )
        if len(vad_segments) == 0:
            return text
        for vad_segment in vad_segments:
            start_ms, end_ms = vad_segment
            segment_audio = self.audio_processor.extract_segment(
                audio_segment, start_ms, end_ms
            )
            text += self._get_asr_text(segment_audio)
        return text

    def _get_asr_text(self, audio_segment: AudioSegment) -> str:
        self.model_manager.ensure_loaded()
        assert self.model_manager.asr_model is not None
        result = self.model_manager.asr_model.generate(
            input=audio_segment, sample_rate=self.audio_processor.sample_rate
        )  # type: ignore
        return "".join(result[0]["text"].split())

    def get_punc_text(self, text: str) -> str:
        self.model_manager.ensure_loaded()
        assert self.model_manager.punc_model is not None
        result = self.model_manager.punc_model.generate(input=text)
        return result[0]["text"]

    def get_fazh_result(
        self, audio_path: Path, text_path: Path
    ) -> List[Dict[str, Any]]:
        self.model_manager.ensure_loaded()
        assert self.model_manager.fazh_model is not None
        result = self.model_manager.fazh_model.generate(
            input=(str(audio_path), str(text_path)), data_type=("sound", "text")
        )
        return result

    def process_vad_segment(
        self,
        audio_data: AudioSegment,
        vad_segment: VadSegment,
    ) -> str:
        """处理单个VAD片段，返回带标点的文本"""
        start_ms, end_ms = vad_segment
        segment_audio = self.audio_processor.extract_segment(
            audio_data, start_ms, end_ms
        )
        asr_text = self.get_asr_text(segment_audio)
        if len(asr_text) == 0:
            return ""
        punc_text = self.get_punc_text(asr_text)
        return punc_text

    def fix_word_timestamps(
        self,
        word_timestamps: List[WordTimestamp],
        vad_segment: VadSegment,
    ) -> List[WordTimestamp]:
        """修正FA-ZH返回的单词时间戳

        TODO: 直接取平均值不能覆盖所有情况，需要更复杂的算法，考虑标点符号的影响和VAD时间戳的准确性
        """
        if not word_timestamps:
            return []

        vad_start, vad_end = vad_segment

        # 计算首尾时间差异的平均调整量
        first_word_start_diff = word_timestamps[0][1][0] - vad_start
        last_word_end_diff = word_timestamps[-1][1][1] - vad_end
        avg_adjustment = (
            first_word_start_diff + last_word_end_diff
        ) / 2 + 10  # 微调漂移，猜测可能和标点有关

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

    def process_long_segment(
        self,
        audio_data: AudioSegment,
        vad_segment: VadSegment,
        punc_text: str,
        temp_dir: Path,
    ) -> List[SentenceTimestamp]:
        """处理长音频片段，使用FA-ZH进行细粒度对齐"""
        start_ms, end_ms = vad_segment

        # 保存临时音频文件
        temp_audio_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.wav"
        segment_audio = self.audio_processor.extract_segment(
            audio_data, start_ms, end_ms
        )
        self.audio_processor.save_audio(segment_audio, temp_audio_path)

        # 保存临时文本文件（移除标点）
        temp_text_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.txt"
        plain_text = self.remove_punctuation(punc_text)
        temp_text_path.write_text(plain_text, encoding="utf-8")

        # 获取FA-ZH结果
        fazh_result = self.get_fazh_result(temp_audio_path, temp_text_path)
        fazh_text = fazh_result[0]["text"].split()
        fazh_timestamps = fazh_result[0]["timestamp"]

        #! FA-ZH 的时间戳是相对于片段的，需要转换为绝对时间戳
        #! 同时展开多字符的 token（如 "VY"）
        absolute_timestamps: List[WordTimestamp] = []
        
        # FA-ZH 可能将多个字符识别为一个 token，需要展开
        for token, timestamp_pair in zip(fazh_text, fazh_timestamps):
            absolute_start = timestamp_pair[0] + start_ms
            absolute_end = timestamp_pair[1] + start_ms
            
            if len(token) > 1:
                # 多字符 token，平均分配时间戳给每个字符
                duration_per_char = (absolute_end - absolute_start) / len(token)
                for i, char in enumerate(token):
                    char_start = int(absolute_start + i * duration_per_char)
                    char_end = int(absolute_start + (i + 1) * duration_per_char)
                    absolute_timestamps.append((char, [char_start, char_end]))
            else:
                # 单字符，直接使用
                absolute_timestamps.append((token, [absolute_start, absolute_end]))

        # 修正标点
        fixed_timestamps = self.fix_word_timestamps(absolute_timestamps, vad_segment)

        # 强制对齐
        sentences = self.serialize_sentences(punc_text)
        aligned_timestamps = self.align_sentence_timestamps(fixed_timestamps, sentences)

        return aligned_timestamps

    def __call__(
        self,
        audio_path: Path,
        output_dir: Path,
        save_segments: bool = True,
    ) -> ProcessingResult:
        """处理音频文件，生成SRT字幕"""
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
        #! 转换采样率
        if sample_rate != self.audio_processor.sample_rate:
            audio_data = self.audio_processor.resample_audio(
                audio_data,
                original_rate=sample_rate,
                target_rate=self.audio_processor.sample_rate,
            )
            sample_rate = self.audio_processor.sample_rate
        #! 获取VAD分段
        vad_segments = self.get_vad_segments(audio_data, sample_rate=sample_rate)
        vad_punc_texts: List[str] = []
        for vad_segment in vad_segments:
            punc_text = self.process_vad_segment(audio_data, vad_segment)
            vad_punc_texts.append(punc_text)
        all_sentence_timestamps: List[SentenceTimestamp] = []
        all_char_timestamps: List[WordTimestamp] = []  # 收集所有字符的时间戳

        for vad_segment, punc_text in zip(vad_segments, vad_punc_texts):
            invalid_texts = []
            if not self.is_valid_text(punc_text, invalid_texts):
                # 空文本或无效文本跳过
                continue

            start_ms, end_ms = vad_segment
            duration_ms = end_ms - start_ms

            # 长片段使用FA-ZH细化
            if duration_ms > 3000:
                sentence_timestamps = self.process_long_segment(
                    audio_data, vad_segment, punc_text, output_dir
                )
                all_sentence_timestamps.extend(sentence_timestamps)

                # 从句子时间戳中提取字符级时间戳
                char_timestamps = self._extract_char_timestamps_from_sentences(
                    sentence_timestamps
                )
                all_char_timestamps.extend(char_timestamps)
            else:
                # 短片段直接使用VAD时间戳
                sentences = self.serialize_sentences(punc_text)
                if sentences:
                    all_sentence_timestamps.append((sentences[0], [start_ms, end_ms]))
                    # 为短片段生成平均分配的字符时间戳
                    char_timestamps = self._generate_uniform_char_timestamps(
                        sentences[0], start_ms, end_ms
                    )
                    all_char_timestamps.extend(char_timestamps)
        audio_path_name = audio_path.stem
        audio_name = [part for part in audio_path_name.split("/") if part][-1]
        srt_path = output_dir / f"{audio_name}.srt"
        self.generate_srt_file(all_sentence_timestamps, srt_path)

        if save_segments:
            self._save_final_audio_segments(
                output_dir, audio_data, all_sentence_timestamps
            )

        return ProcessingResult(
            segments=all_sentence_timestamps,
            vad_segments=vad_segments,
            char_timestamps=all_char_timestamps,
            metadata={
                "total_sentences": len(all_sentence_timestamps),
                "total_vad_segments": len(vad_segments),
                "audio_duration_ms": int(len(audio_data) * 1000 / sample_rate),
                "sample_rate": sample_rate,
            },
        )

    def generate_srt_file(
        self,
        sentence_timestamps: List[SentenceTimestamp],
        output_path: Path,
    ) -> None:
        """生成SRT字幕文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, (text, timestamp) in enumerate(sentence_timestamps, 1):
                start_time_str = self.ms_to_srt_time(timestamp[0])
                end_time_str = self.ms_to_srt_time(timestamp[1])
                f.write(f"{idx}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{text}\n\n")

    def _save_final_audio_segments(
        self,
        output_dir: Path,
        audio_data: AudioSegment,
        sentence_timestamps: List[SentenceTimestamp],
    ) -> None:
        """保存最终句子音频片段"""
        final_dir = output_dir / "final_segments"
        final_dir.mkdir(exist_ok=True)

        for idx, (text, timestamp) in enumerate(sentence_timestamps, 1):
            start_ms, end_ms = timestamp
            segment = self.audio_processor.extract_segment(audio_data, start_ms, end_ms)

            safe_text = text[:20].replace(" ", "_").replace("/", "_")
            output_path = final_dir / f"segment_{idx:04d}_{safe_text}.wav"

            self.audio_processor.save_audio(
                segment,
                output_path,
            )

    @staticmethod
    def ms_to_srt_time(ms: TimestampMs) -> str:
        total_ms = int(round(ms))
        hours = total_ms // 3600000
        remaining = total_ms % 3600000
        minutes = remaining // 60000
        remaining = remaining % 60000
        seconds = remaining // 1000
        milliseconds = remaining % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def serialize_sentences(self, text: str) -> List[str]:
        """将带标点的文本切分为句子列表"""
        for delimiter in self.sentence_delimiters:
            text = text.replace(delimiter, "|")
        sentences = [s.strip() for s in text.split("|") if s.strip()]
        return sentences

    def remove_punctuation(self, text: str) -> str:
        """移除标点符号"""
        for delimiter in self.sentence_delimiters:
            text = text.replace(delimiter, "")
        return text

    @staticmethod
    def is_valid_text(text: str, invalid_list: List[str]) -> bool:
        """检查文本是否有效"""
        return len(text) > 0 and text not in invalid_list

    def analyze_punctuation(self, text: str) -> Dict[str, int]:
        """统计句子中对应标点符号总数"""
        punctuation_counts: Dict[str, int] = {}
        for char in text:
            if char in self.sentence_delimiters:
                if char not in punctuation_counts:
                    punctuation_counts[char] = 0
                punctuation_counts[char] += 1
        return punctuation_counts

    def _extract_char_timestamps_from_sentences(
        self, sentence_timestamps: List[SentenceTimestamp]
    ) -> List[WordTimestamp]:
        """从句子时间戳中提取字符级时间戳

        对于长音频片段，FA-ZH已经提供了字符级时间戳，
        这里需要根据句子重新生成包含标点的字符时间戳
        """
        char_timestamps: List[WordTimestamp] = []

        for sentence, timestamp in sentence_timestamps:
            start_ms, end_ms = timestamp
            sentence_len = len(sentence)

            if sentence_len == 0:
                continue

            # 为句子中的每个字符平均分配时间
            duration_per_char = (end_ms - start_ms) / sentence_len

            for i, char in enumerate(sentence):
                char_start = int(start_ms + i * duration_per_char)
                char_end = int(start_ms + (i + 1) * duration_per_char)
                char_timestamps.append((char, [char_start, char_end]))

        return char_timestamps

    def _generate_uniform_char_timestamps(
        self, text: str, start_ms: TimestampMs, end_ms: TimestampMs
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


def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    # AUDIO_DIR = ROOT_DIR / "outputs" / "audio"
    # TEST_AUDIO = (
    #     AUDIO_DIR / "Feishu20251105-100728.wav"
    # )  # AUDIO_DIR / "sep_6be05f9dd9a448bda18e856b76b8fa10_vocals.wav"
    # print(TEST_AUDIO.exists())
    TEST_AUDIO = Path("/home/hsiangnianian/GitProject/deving/HsiangNianian/funasr-api/audio/sep_f747dde8dfd542b4a5c22156456a0eff_vocals.wav")
    OUTPUT_DIR = ROOT_DIR / "outputs"

    generator = SRTGenerator()
    generator.load_models()

    result = generator(audio_path=TEST_AUDIO, output_dir=OUTPUT_DIR)
    print(result)


if __name__ == "__main__":
    main()
