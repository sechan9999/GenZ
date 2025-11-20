"""
Gen Z Agent - Multi-Agent Invoice Automation System
한국 선거 데이터 분석 및 청구서 자동화 시스템

CrewAI + Anthropic Claude를 활용한 다중 에이전트 시스템
"""

__version__ = "2.0.0"
__author__ = "Gen Z Agent Team"

from .crew import create_electoral_analysis_crew, run_electoral_analysis
from .config import Config

__all__ = [
    "create_electoral_analysis_crew",
    "run_electoral_analysis",
    "Config",
]
