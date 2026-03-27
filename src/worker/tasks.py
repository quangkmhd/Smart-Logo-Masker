import os
import time
import logging
from src.worker.celery_app import celery_app
from src.services.video_processor import video_processor
from src.core.config import settings

from src.schemas.task import ProcessingOptions

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, filename: str, options: dict = None):
    self.update_state(state='PROGRESS', meta={'progress': 0})
    
    # Parse options
    proc_options = ProcessingOptions(**(options or {}))
    
    input_path = os.path.join(settings.UPLOAD_DIR, filename)
    output_filename = f"masked_{filename}"
    output_path = os.path.join(settings.RESULT_DIR, output_filename)
    
    logger.info(f"Processing video with options {proc_options}: {filename} -> {output_filename}")
    
    try:
        success = video_processor.process(input_path, output_path, proc_options)
        if success:
            logger.info(f"Successfully processed {filename}")
            return {
                "status": "COMPLETED",
                "result_filename": output_filename,
                "result_path": output_path
            }
        else:
            raise Exception("Processing failed")
            
    except Exception as e:
        logger.error(f"Error in task: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise e
