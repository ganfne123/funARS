"""workflow.run 的真实 VAD 链路测试。"""

import os
import tempfile
import unittest
from pathlib import Path

from src.main import ProcessingResult, run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_PATH = PROJECT_ROOT / "ttsmaker-file-2026-8-26-0-9-55.mp3"


@unittest.skipUnless(
    os.getenv("RUN_FUNASR_INTEGRATION") == "1",
    "真实 FunASR 工作流测试默认跳过；设置 RUN_FUNASR_INTEGRATION=1 后运行。",
)
class WorkflowRunTests(unittest.TestCase):
    def test_run_returns_vad_segments(self) -> None:
        """run 应完成读取、16 kHz 重采样、VAD 和结果封装。"""
        if not AUDIO_PATH.is_file():
            self.skipTest(f"测试音频不存在: {AUDIO_PATH}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            result = run(
                "workflow-test",
                audio_path=AUDIO_PATH,
                output_dir=output_dir,
                save_segments=False,
            )

        print("run() VAD 分段:", result.vad_segments)
        print("run() 元数据:", result.metadata)

        self.assertIsInstance(result, ProcessingResult)
        self.assertTrue(result.vad_segments, "run() 没有返回 VAD 分段")
        self.assertEqual(result.metadata["sample_rate"], 16000)
        self.assertEqual(
            result.metadata["total_vad_segments"],
            len(result.vad_segments),
        )
        self.assertGreater(result.metadata["audio_duration_ms"], 0)

        for start_ms, end_ms in result.vad_segments:
            self.assertGreaterEqual(start_ms, 0)
            self.assertGreater(end_ms, start_ms)


if __name__ == "__main__":
    unittest.main()
