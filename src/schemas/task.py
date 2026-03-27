from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ProcessingOptions(BaseModel):
    conf: float = Field(0.75, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    blur_intensity: int = Field(99, ge=1, le=201)  # Must be odd for GaussianBlur
    mask_mode: str = Field("blur", pattern="^(blur|solid)$")
    target_classes: Optional[List[int]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = 0
    result: Optional[Dict[str, Any]] = None
    options: Optional[ProcessingOptions] = None
