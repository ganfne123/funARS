"""项目级回归测试：模型封装、音频工作流和 FastAPI 接口。

全部模型调用均使用 ``FakeModels``，因此不下载模型、不依赖 GPU：

    UV_CACHE_DIR=/tmp/funasr-api-uv-cache uv run python -m unittest discover \
        -s tests -p 'test_project.py' -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

import src.app as app_module
import src.main as main_module
import src.workflow as workflow_module
from src.Entities import ProcessingResult, TaskInfo, TaskStatus, tasks
from src.models import FunASRModels


class FakeModels:
    """以确定性结果代替 FunASR，并记录是否走过强制对齐。"""

    def __init__(self, *, vad_end_ms: int = 1000, ready: bool = True) -> None:
        self.vad_end_ms = vad_end_ms
        self.ready = ready
        self.align_called = False

    def is_ready(self) -> bool:
        return self.ready

    def vad(self, audio_data: np.ndarray, sample_rate: int) -> list[dict]:
        return [{"value": [[0, self.vad_end_ms]]}]

    def asr(self, audio_data: np.ndarray, sample_rate: int) -> list[dict]:
        return [{"text": "你 好 世 界"}]

    def punctuate(self, text: str) -> list[dict]:
        return [{"text": "你好，世界。"}]

    def align(
        self,
        audio_data: np.ndarray,
        text: str,
        sample_rate: int,
    ) -> list[dict]:
        self.align_called = True
        return [
            {
                "text": "你 好 世 界",
                "timestamp": [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000]],
            }
        ]


class FakeUpload:
    """只实现路由所需的 UploadFile 最小接口。"""

    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self.content = content

    async def read(self) -> bytes:
        return self.content


class ModelWrapperTests(unittest.TestCase):
    def test_cache_readiness_and_local_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            models = FunASRModels(cache_dir=cache_dir)
            self.assertFalse(models.is_ready())

            for cache_name in models.model_cache_names.values():
                model_dir = cache_dir / cache_name
                model_dir.mkdir()
                (model_dir / "config.yaml").touch()
                (model_dir / "model.pt").touch()

            self.assertTrue(models.is_ready())
            expected_path = cache_dir / models.model_cache_names["fsmn-vad"]
            self.assertEqual(models.resolve_model("fsmn-vad"), str(expected_path))

    def test_public_model_methods_delegate_to_generate(self) -> None:
        models = FunASRModels()
        audio = np.zeros(160, dtype=np.float32)

        with patch.object(models, "_generate", return_value=[{"text": "结果"}]) as generate:
            self.assertEqual(models.vad(audio), [{"text": "结果"}])
            self.assertEqual(generate.call_args.args, ("fsmn-vad",))
            self.assertIs(generate.call_args.kwargs["input"], audio)
            self.assertEqual(generate.call_args.kwargs["sample_rate"], 16000)

            generate.reset_mock()
            models.asr(audio, sample_rate=8000)
            self.assertEqual(generate.call_args.args, ("paraformer-zh",))
            self.assertIs(generate.call_args.kwargs["input"], audio)
            self.assertEqual(generate.call_args.kwargs["sample_rate"], 8000)

            generate.reset_mock()
            models.punctuate("你好")
            self.assertEqual(generate.call_args.args, ("ct-punc",))
            self.assertEqual(generate.call_args.kwargs["input"], "你好")

            generate.reset_mock()
            models.align(audio, "你好")
            self.assertEqual(generate.call_args.args, ("fa-zh",))
            self.assertEqual(generate.call_args.kwargs["input"], (audio, "你好"))
            self.assertEqual(generate.call_args.kwargs["data_type"], ("sound", "text"))
            self.assertEqual(generate.call_args.kwargs["sample_rate"], 16000)


class AudioWorkflowTests(unittest.TestCase):
    def _write_audio(self, path: Path, *, seconds: int, sample_rate: int) -> None:
        sf.write(
            path,
            np.zeros(seconds * sample_rate, dtype=np.float32),
            samplerate=sample_rate,
        )

    def test_helper_functions(self) -> None:
        processor = workflow_module.audioProcessor()
        self.assertEqual(processor.ms_to_samples(500), 8000)
        self.assertEqual(
            processor.serialize_sentences("你好，世界。"),
            ["你好", "世界"],
        )
        self.assertEqual(processor.remove_punctuation("你好，世界。"), "你好世界")
        self.assertEqual(
            processor.merge_model_text(
                [{"text": "你 好"}, {"text": "世 界"}],
                remove_whitespace=True,
            ),
            "你好世界",
        )

    def test_short_segment_resamples_and_writes_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "input.wav"
            output_dir = temp_path / "output"
            self._write_audio(audio_path, seconds=1, sample_rate=8000)

            with patch.object(main_module.workflow, "generator", FakeModels()):
                result = main_module.run("task-1", audio_path, output_dir, True)

            self.assertEqual(
                result.segments,
                [("你好", [0, 1000]), ("世界", [0, 1000])],
            )
            self.assertEqual(result.metadata["original_sample_rate"], 8000)
            self.assertEqual(result.metadata["sample_rate"], 16000)
            self.assertEqual(result.metadata["total_vad_segments"], 1)
            self.assertEqual(len(result.char_timestamps), 4)

            timestamp_path = output_dir / "task-1.timestamps.json"
            self.assertTrue(timestamp_path.is_file())
            payload = json.loads(timestamp_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["all_sentence_timestamps"]), 2)

    def test_long_segment_uses_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "input.wav"
            self._write_audio(audio_path, seconds=4, sample_rate=16000)
            fake_models = FakeModels(vad_end_ms=4000)

            with patch.object(main_module.workflow, "generator", fake_models):
                result = main_module.run(
                    "task-2",
                    audio_path,
                    temp_path / "output",
                    False,
                )

            self.assertTrue(fake_models.align_called)
            self.assertEqual([text for text, _ in result.segments], ["你好", "世界"])
            self.assertEqual(result.segments[0][1][0], 0)
            self.assertEqual(result.segments[-1][1][1], 4000)

    def test_background_task_updates_status_and_removes_upload(self) -> None:
        old_tasks = tasks.copy()
        tasks.clear()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                audio_path = temp_path / "upload.wav"
                output_dir = temp_path / "outputs"
                self._write_audio(audio_path, seconds=1, sample_rate=16000)
                tasks["task-3"] = TaskInfo(
                    task_id="task-3",
                    filename="upload.wav",
                    created_at=datetime.now(),
                )
                run_result = ProcessingResult(
                    segments=[("你好", [0, 1000]), ("世界", [0, 1000])],
                    vad_segments=[(0, 1000)],
                    char_timestamps=[],
                    metadata={
                        "total_sentences": 2,
                        "total_vad_segments": 1,
                        "audio_duration_ms": 1000,
                    },
                )

                async def run_inline(
                    function: object,
                    *args: object,
                    **kwargs: object,
                ) -> ProcessingResult:
                    return function(*args, **kwargs)  # type: ignore[operator]

                with (
                    patch.object(main_module.workflow, "generator", FakeModels()),
                    patch.object(main_module, "OUTPUT_DIR", output_dir),
                    patch.object(main_module.workflow, "run", return_value=run_result) as run,
                    patch.object(main_module.asyncio, "to_thread", new=run_inline),
                ):
                    asyncio.run(
                        main_module.process_audio_task("task-3", audio_path, False)
                    )

                self.assertEqual(tasks["task-3"].status, TaskStatus.COMPLETED)
                self.assertFalse(audio_path.exists())
                self.assertEqual(tasks["task-3"].result["total_sentences"], 2)  # type: ignore[index]
                run.assert_called_once_with(
                    task_id="task-3",
                    audio_path=audio_path,
                    output_dir=output_dir / "task-3",
                    save_segments=False,
                )
        finally:
            tasks.clear()
            tasks.update(old_tasks)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.upload_dir = temp_path / "uploads"
        self.output_dir = temp_path / "outputs"
        self.upload_dir.mkdir()
        self.output_dir.mkdir()
        self.old_tasks = tasks.copy()
        tasks.clear()
        self.patches = [
            patch.object(app_module, "UPLOAD_DIR", self.upload_dir),
            patch.object(app_module, "OUTPUT_DIR", self.output_dir),
            patch.object(main_module, "UPLOAD_DIR", self.upload_dir),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        tasks.clear()
        tasks.update(self.old_tasks)
        self.temp_dir.cleanup()

    def test_create_and_query_task(self) -> None:
        calls: list[tuple[str, Path, bool]] = []

        async def complete_task(
            task_id: str,
            audio_path: Path,
            save_segments: bool,
        ) -> None:
            calls.append((task_id, audio_path, save_segments))
            task = tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = {"save_segments": save_segments}

        background_tasks = BackgroundTasks()
        upload = FakeUpload("../../recording.wav", b"test audio")
        with patch.object(app_module, "process_audio_task", complete_task):
            response = asyncio.run(
                app_module.create_task(
                    background_tasks=background_tasks,
                    file=upload,
                    save_segments=True,
                )
            )
            asyncio.run(background_tasks())

        task_id = response.task_id
        self.assertEqual(response.status, TaskStatus.PENDING)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], task_id)
        self.assertTrue(calls[0][2])
        self.assertEqual(tasks[task_id].filename, "recording.wav")

        status_response = asyncio.run(app_module.get_task(task_id))
        self.assertEqual(status_response.status, TaskStatus.COMPLETED)
        with self.assertRaises(HTTPException) as context:
            asyncio.run(app_module.get_task("missing"))
        self.assertEqual(context.exception.status_code, 404)

    def test_download_requires_completed_task_and_returns_timestamp_file(self) -> None:
        with self.assertRaises(HTTPException) as context:
            asyncio.run(app_module.download_file("missing"))
        self.assertEqual(context.exception.status_code, 404)

        tasks["pending"] = TaskInfo(
            task_id="pending",
            filename="audio.wav",
            created_at=datetime.now(),
        )
        with self.assertRaises(HTTPException) as context:
            asyncio.run(app_module.download_file("pending"))
        self.assertEqual(context.exception.status_code, 400)

        tasks["completed"] = TaskInfo(
            task_id="completed",
            status=TaskStatus.COMPLETED,
            filename="audio.wav",
            created_at=datetime.now(),
        )
        timestamp_dir = self.output_dir / "completed"
        timestamp_dir.mkdir()
        expected_file = timestamp_dir / "completed.timestamps.json"
        expected_file.write_text(
            json.dumps({"all_sentence_timestamps": []}),
            encoding="utf-8",
        )

        response = asyncio.run(app_module.download_file("completed"))
        self.assertIsInstance(response, FileResponse)
        self.assertEqual(Path(response.path), expected_file.resolve())
        self.assertEqual(response.filename, "completed.timestamps.json")


class MainEntryPointTests(unittest.TestCase):
    def test_module_run_delegates_to_workflow_instance(self) -> None:
        expected_result = ProcessingResult([], [], [], {})
        with patch.object(main_module.workflow, "run", return_value=expected_result) as run:
            result = main_module.run("task-4", Path("input.wav"), Path("output"), True)

        self.assertIs(result, expected_result)
        run.assert_called_once_with("task-4", Path("input.wav"), Path("output"), True)


if __name__ == "__main__":
    unittest.main()
