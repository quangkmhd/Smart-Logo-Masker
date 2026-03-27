from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import os
import uuid
import shutil
import json
from src.core.config import settings
from src.schemas.task import TaskResponse, TaskStatus, ProcessingOptions
from src.worker.celery_app import celery_app
from src.worker.tasks import process_video_task

router = APIRouter()

@router.post("/process", response_model=TaskResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    options: str = Form("{}")  # As JSON string from Gradio
):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    try:
        opt_dict = json.loads(options)
        proc_options = ProcessingOptions(**opt_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid processing options: {e}")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger background task with options
    task = process_video_task.delay(unique_filename, proc_options.dict())
    
    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message="Video uploaded and processing started"
    )

@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "progress": 0,
        "result": None,
        "options": None
    }
    
    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["result"] = {"error": str(task_result.info)}
    elif task_result.status == "PROGRESS":
        response["progress"] = task_result.info.get("progress", 0)
        
    return response

@router.get("/download/{task_id}")
async def download_result(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    if task_result.status != "SUCCESS":
        raise HTTPException(status_code=404, detail="Task not finished or failed")
    
    result_data = task_result.result
    file_path = result_data.get("result_path")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
        
    return FileResponse(
        path=file_path,
        filename=result_data.get("result_filename"),
        media_type='video/mp4'
    )
