from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
import json
from pathlib import Path
from datetime import datetime

from src.main import create_upload_task, process_audio_task
from src.Entities import *
from fastapi.responses import FileResponse



app = FastAPI()


@app.post(
    "/api/v1/tasks",
    response_model=CreateTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="音频文件"),
    save_segments: bool = Query(default=False, description="是否保存音频片段"),
) -> CreateTaskResponse:
    task_id, upload_path = await create_upload_task(file)

    task = tasks[task_id]
    background_tasks.add_task(
            process_audio_task,
            task_id=task_id,
            audio_path=upload_path,
            save_segments=save_segments,
        )
    

    return CreateTaskResponse(task_id=task_id, status=task.status)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str) -> TaskInfo:
    """返回指定任务的当前状态及完成后的处理结果。"""
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return task



@app.get("/api/v1/download/{task_id}/timestamps")
async def download_file(task_id: str):
    """下载指定任务的时间戳文件。"""
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务尚未完成，无法下载文件")
    file_path = (
        OUTPUT_DIR
        / task_id
        / f"{task_id}.timestamps.json"
    ).resolve()

    return FileResponse(
        path=file_path,
        filename=file_path.name,  # 浏览器下载时显示的文件名
        media_type="application/octet-stream",
    )
