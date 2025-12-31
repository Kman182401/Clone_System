"""
File-Window Library Modules

This package contains intelligent analysis tools for the clone system:

- pii_detector: Comprehensive PII detection and redaction
- smart_file_analyzer: Intelligent file dependency and relevance analysis
- ai_relevance_detector: AI-powered file relevance detection using Ollama
"""

from .pii_detector import (
    PIIDetector,
    PIIMatch,
    PIICategory,
    Sensitivity,
    scan_text,
    redact_text,
    scan_file,
)

from .smart_file_analyzer import (
    SmartFileAnalyzer,
    FileCategory,
    FileInfo,
    AnalysisResult,
    analyze_codebase,
    get_trading_system_files,
    generate_clone_config,
)

# AI detection (optional - requires Ollama)
try:
    from .ai_relevance_detector import (
        AIRelevanceDetector,
        Relevance,
        FileAnalysis,
        check_file_relevance,
        run_baseline_scan,
    )
    HAS_AI_DETECTOR = True
except ImportError:
    HAS_AI_DETECTOR = False
    AIRelevanceDetector = None
    Relevance = None
    FileAnalysis = None
    check_file_relevance = None
    run_baseline_scan = None

__all__ = [
    # PII Detection
    'PIIDetector',
    'PIIMatch',
    'PIICategory',
    'Sensitivity',
    'scan_text',
    'redact_text',
    'scan_file',

    # Smart File Analysis
    'SmartFileAnalyzer',
    'FileCategory',
    'FileInfo',
    'AnalysisResult',
    'analyze_codebase',
    'get_trading_system_files',
    'generate_clone_config',

    # AI Relevance Detection
    'AIRelevanceDetector',
    'Relevance',
    'FileAnalysis',
    'check_file_relevance',
    'run_baseline_scan',
    'HAS_AI_DETECTOR',
]
