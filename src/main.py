import torch
import torchaudio.functional as AF
from typing import List
from pathlib import Path
import asyncio
from fastapi import HTTPException,UploadFile
import uuid


from src.models import FunASRModels
from src.Entities import *
from src.workflow import audioProcessor



generator = FunASRModels(device="cuda:0")
audio_processor= audioProcessor()
def run(
    task_id: str,
    audio_path: Path,
    output_dir: Path,
    save_segments: bool,
) -> ProcessingResult:
    """读取音频基础信息；后续步骤将在这里继续完成模型推理。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_data, sample_rate = audio_processor.prepare_audio(audio_path)
    original_sample_rate = sample_rate
    target_sample_rate = 16000

    if sample_rate != target_sample_rate:
        # NumPy 音频数组转换成 PyTorch 张量
        audio_tensor = torch.from_numpy(audio_data)
        audio_data = AF.resample(
            audio_tensor,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        ).numpy()
        sample_rate = target_sample_rate

    audio_duration_ms = int(len(audio_data) * 1000 / sample_rate)

    vad_result = generator.vad(audio_data, sample_rate=sample_rate)
    vad_segments: List[VadSegment] = []
    for result in vad_result:
        for start_ms, end_ms in result.get("value", []):
            vad_segments.append((int(start_ms), int(end_ms)))

    # 汇总所有 VAD 片段产生的句子级结果。
    all_sentence_timestamps: List[SentenceTimestamp] = []
    all_char_timestamps: List[WordTimestamp] = []

    # 逐个处理 VAD 片段：统一单位
    for start_ms, end_ms in vad_segments:
        start_sample = max(0, start_ms * sample_rate // 1000)
        end_sample = min(len(audio_data), end_ms * sample_rate // 1000)
        if end_sample <= start_sample:
            continue

        # 提取文本并清洗
        segment_audio = audio_data[start_sample:end_sample]
        asr_result = generator.asr(segment_audio, sample_rate=sample_rate)
        asr_text =  audio_processor.merge_model_text(asr_result, remove_whitespace=True)
        if not asr_text:
            continue

        # 恢复标点。
        punc_result = generator.punctuate(asr_text)
        punc_text =  audio_processor.merge_model_text(punc_result)
        if not punc_text:
            continue

        vad_segment = (start_ms, end_ms)
        duration_ms = end_ms - start_ms
        if duration_ms > 3000:
            # FA-ZH 的时间戳相对当前 VAD 片段，因此传入片段音频。
            sentence_timestamps = process_long_segment(
                segment_audio,
                vad_segment,
                punc_text,
                output_dir,
            )
        else:
            sentences =  audio_processor.serialize_sentences(punc_text)
            sentence_timestamps = [
                (sentence, [start_ms, end_ms]) for sentence in sentences
            ]

        all_sentence_timestamps.extend(sentence_timestamps)
        for sentence, (sentence_start, sentence_end) in sentence_timestamps:
            all_char_timestamps.extend(
                    audio_processor.generate_uniform_char_timestamps(
                    sentence,
                    sentence_start,
                    sentence_end,
                )
            )
            audio_processor.write_timestamps_file(
                all_sentence_timestamps,
                output_dir / f"{task_id}.timestamps.json",)

    return ProcessingResult(
        segments=all_sentence_timestamps,
        vad_segments=vad_segments,
        char_timestamps=all_char_timestamps,
        metadata={
            "total_sentences": len(all_sentence_timestamps),
            "total_vad_segments": len(vad_segments),
            "audio_duration_ms": audio_duration_ms,
            "original_sample_rate": original_sample_rate,
            "sample_rate": sample_rate,
            "save_segments": save_segments,
        },
    )


async def process_audio_task(
    task_id: str,
    audio_path: Path,
    save_segments: bool,
):
    """后台处理音频任务"""


    task = tasks[task_id]

    try:
        task.status = TaskStatus.PROCESSING
        print(f"开始处理任务: {task_id}")
        output_dir = OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # 在线程中执行同步函数 run(...)。当前协程用 await 等待它完成。等待期间，FastAPI 仍可以处理其他请求
        result = await asyncio.to_thread(
            run,
            task_id=task_id,
            audio_path=audio_path,
            output_dir=output_dir,
            save_segments=save_segments,
        )


        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = {
            "total_sentences": result.metadata["total_sentences"],
            "total_vad_segments": result.metadata["total_vad_segments"],
            "audio_duration_ms": result.metadata["audio_duration_ms"],
            "segments": [
                {
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
                for text, (start_ms, end_ms) in result.segments
            ],
            "save_segments": save_segments,
        }

        print(f"任务完成: {task_id}, 生成 {result.metadata['total_sentences']} 个字幕")

    except Exception as e:
        print(f"任务处理失败: {task_id}, 错误: {e}")
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error_message = str(e)

    finally:
        if audio_path.exists():
            audio_path.unlink()



def process_long_segment(
            audio_data: AudioSegment,
            vad_segment: VadSegment,
            punc_text: str,
            temp_dir: Path,
        ) -> List[SentenceTimestamp]:
            """处理长音频片段，使用FA-ZH进行细粒度对齐"""
            start_ms, end_ms = vad_segment

            # 保存临时文本文件（移除标点）
            temp_text_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.txt"
            plain_text =  audio_processor.remove_punctuation(punc_text)
            temp_text_path.write_text(plain_text, encoding="utf-8")

            # 获取FA-ZH结果
            fazh_result = generator.align(audio_data, plain_text, sample_rate=16000)
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

            # 修正FA-ZH返回的单词时间戳
            fixed_timestamps =  audio_processor.fix_word_timestamps(absolute_timestamps, vad_segment)

            # 按照标点切分
            sentences =  audio_processor.serialize_sentences(punc_text)

            aligned_timestamps: list[tuple[str, list[int]]] =  audio_processor.align_sentence_timestamps(fixed_timestamps, sentences)

            return aligned_timestamps
def extract_segment(
            self,
            audio_data: AudioSegment, start_ms: TimestampMs, end_ms: TimestampMs
        ) -> AudioSegment:
            """根据采样点提取音频片段"""
            start_sample = self.audio_processor.ms_to_samples(start_ms)
            end_sample = self.audio_processor.ms_to_samples(end_ms)
            return audio_data[start_sample:end_sample]

async def create_upload_task(file: UploadFile) -> Tuple[str, Path]:
    """上传音频并创建后台处理任务。"""
    task_id = str(uuid.uuid4())
    filename = Path(file.filename or "audio.wav").name
    upload_dir = UPLOAD_DIR / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / filename

    with upload_path.open("wb") as f:
        content = await file.read()
        f.write(content)


    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        filename=filename,
        created_at=datetime.now(),
    )
    tasks[task_id] = task_info
    return task_id, upload_path


