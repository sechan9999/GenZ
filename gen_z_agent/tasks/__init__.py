"""
Task definitions for Gen Z Agent CrewAI workflow
"""

from .electoral_tasks import (
    create_extraction_task,
    create_validation_task,
    create_analysis_task,
    create_report_task,
    create_notification_task,
)

__all__ = [
    "create_extraction_task",
    "create_validation_task",
    "create_analysis_task",
    "create_report_task",
    "create_notification_task",
]
