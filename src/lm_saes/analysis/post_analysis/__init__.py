"""Post-analysis processors for different SAE types.

This module provides post-processing functionality for analysis results,
allowing different SAE implementations to customize how their analysis
results are formatted and processed.
"""

# Import base classes and registration functions first
from .base import PostAnalysisProcessor, get_post_analysis_processor, register_post_analysis_processor

# Import all processors to trigger their registration
from .clt import CLTPostAnalysisProcessor
from .crosscoder import CrossCoderPostAnalysisProcessor
from .generic import GenericPostAnalysisProcessor
from .lorsa import LorsaPostAnalysisProcessor

__all__ = [
    "PostAnalysisProcessor",
    "register_post_analysis_processor",
    "get_post_analysis_processor",
    "CrossCoderPostAnalysisProcessor",
    "GenericPostAnalysisProcessor",
    "LorsaPostAnalysisProcessor",
    "CLTPostAnalysisProcessor",
]
