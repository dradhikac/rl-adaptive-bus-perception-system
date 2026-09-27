"""
src/video
===============================================================================
Video ingestion, frame synchronization, and temporal scene analysis pipeline.
===============================================================================
"""

from .video_pipeline import (
    VideoIngestionPipeline,
    VideoJobStore,
    process_multi_camera_videos,
    get_global_job_store,
)

__all__ = [
    "VideoIngestionPipeline",
    "VideoJobStore",
    "process_multi_camera_videos",
    "get_global_job_store",
]
