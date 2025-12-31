#!/usr/bin/env python3
"""
ai_relevance_detector.py - AI-Powered File Relevance Detection using Ollama

Uses a local LLM (via Ollama) to intelligently determine if files are related
to the SPY vertical spread options trading system.

Features:
- Local LLM inference (no API costs, fast, private)
- Persistent baseline cache to avoid redundant analysis
- Content hash-based change detection
- Two-pass analysis: keyword pre-scoring + AI verification
- Confidence calibration based on content analysis
- Few-shot examples for improved accuracy

Author: Clone System v2.3.0
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama3.2:1b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
BASELINE_FILE = Path(__file__).parent.parent / ".ai_baseline.json"
MAX_CONTENT_LENGTH = 6000  # Characters to send to model (leave room for prompt)

# File extensions to analyze
ANALYZABLE_EXTENSIONS = {
    '.py', '.pyi', '.js', '.ts', '.jsx', '.tsx',
    '.yaml', '.yml', '.toml', '.json', '.ini', '.cfg', '.conf',
    '.sh', '.bash', '.zsh',
    '.sql', '.md', '.txt', '.rst',
}

# Directories to skip (exact match)
SKIP_DIRECTORIES = {
    '__pycache__', '.git', '.venv', 'venv', 'node_modules',
    '.mypy_cache', '.pytest_cache', '.tox', 'dist', 'build',
    'egg-info', '.eggs', '.idea', '.vscode',
    '.wfo_venv', 'ml_env', 'env', '.env',
}

# Pattern-based directory exclusion (if dir name contains any of these)
SKIP_DIR_PATTERNS = {'venv', 'site-packages', 'dist-info', 'egg-info', 'node_modules'}

# Files to skip
SKIP_FILES = {
    '.gitignore', '.dockerignore', 'requirements.txt', 'setup.py',
    'pyproject.toml', 'package.json', 'package-lock.json',
    'Makefile', 'Dockerfile', '.env.example', 'LICENSE', 'CHANGELOG.md',
}

# ─────────────────────────────────────────────────────────────────────────────
# Keyword Scoring System (Pre-AI Filter)
# ─────────────────────────────────────────────────────────────────────────────

# Strong positive indicators (trading system specific)
STRONG_POSITIVE_KEYWORDS = {
    # Core trading concepts
    'spy': 3.0, 'vertical_spread': 4.0, 'vertical spread': 4.0,
    'bull_call': 3.5, 'bear_put': 3.5, 'bull_put': 3.5, 'bear_call': 3.5,
    'iron_condor': 3.0, 'credit_spread': 3.0, 'debit_spread': 3.0,

    # Greeks
    'delta': 1.5, 'gamma': 1.5, 'theta': 1.5, 'vega': 1.5,
    'black_scholes': 2.5, 'black-scholes': 2.5, 'bs_greeks': 3.0,
    'implied_volatility': 2.0, 'iv_rank': 2.5,

    # IBKR specific (multiple variations)
    'ib_insync': 3.5, 'ibkr': 3.0, 'interactive_brokers': 3.0,
    'interactive brokers': 3.0, 'ib gateway': 3.0, 'ib_gateway': 3.0,
    'tws': 2.0, 'ibgateway': 3.0, 'ibg': 2.5, 'trader workstation': 2.5,
    'reqmktdata': 2.5, 'placeorder': 2.5, 'ib.connect': 3.0,
    'ibkr gateway': 3.0, 'ibkr_health': 3.0, 'connection_health': 2.0,

    # Market data
    'databento': 3.0, 'opra': 2.5, 'polygon': 2.0,
    'market_data': 1.5, 'option_chain': 2.5, 'options_data': 2.5,

    # Trading operations
    'position_size': 2.0, 'risk_management': 2.0, 'entry_signal': 2.5,
    'exit_signal': 2.5, 'trade_execution': 2.5, 'order_management': 2.0,
    'backtest': 1.5, 'walk_forward': 2.5,

    # ML/Sentiment for trading
    'finbert': 3.0, 'sentiment_pipeline': 2.5, 'news_sentiment': 2.5,
    'trading_signal': 2.5, 'ml_model': 1.5, 'trade_prediction': 2.5,

    # Project-specific
    'omega': 2.0, 'm5_trader': 3.0, 'clone_system': 2.0,
    'ai trading': 2.5, 'trading system': 2.5,
}

# Moderate positive indicators
MODERATE_POSITIVE_KEYWORDS = {
    'options': 1.0, 'trading': 1.0, 'strike': 1.5, 'expiry': 1.5,
    'expiration': 1.0, 'premium': 1.0, 'contract': 0.8,
    'underlying': 1.0, 'put': 0.7, 'call': 0.7,
    'broker': 0.8, 'execution': 0.8, 'fill': 0.8,
    'pnl': 1.2, 'profit': 0.6, 'loss': 0.5,
    'ingest': 0.8, 'monitor': 0.6, 'heartbeat': 0.8,
    'sentiment': 1.0, 'news': 0.6, 'finbert': 1.5,
    # Options strategy terms
    'buy_strike': 2.0, 'sell_strike': 2.0, 'legspec': 2.0,
    'combo_router': 2.0, 'strategy_template': 2.0,
    'frozen': 0.5, 'dataclass': 0.3,
    # Data/IO for trading
    'parquet': 1.0, 'market_data': 1.5, 'ticker': 1.5, 'tickers': 1.5,
    'tradable': 1.5, 'canonical_name': 1.0, 'entity_linking': 1.5,
    'finance_news': 2.0, 'finance news': 2.0,
}

# Strong negative indicators (third-party libraries, unrelated code)
STRONG_NEGATIVE_KEYWORDS = {
    # Generic libraries (NOT trading specific)
    'matplotlib': -2.0, 'pyplot': -1.5, 'mpl_toolkits': -3.0,
    'psutil': -2.5, 'pillow': -2.5, 'pil': -2.0,
    'six.moves': -2.5, 'kiwisolver': -3.0,
    'eventkit': -2.0, 'nest_asyncio': -1.5,

    # Compatibility libraries
    'python 2 and 3 compatibility': -4.0, 'py2': -2.0, 'py3': -1.0,
    'compatibility library': -3.0, 'compatibility utilities': -3.0,
    'iteritems': -1.5, 'movedmodule': -2.0,

    # Test files for third-party libs
    'test_art3d': -3.0, 'test_axes': -3.0, 'test_legend': -3.0,
    'test_unicode': -2.5, 'test_osx': -3.0, 'test_bsd': -3.0,
    'test_linux': -2.5, 'test_windows': -2.5, 'test_system': -2.0,
    'test_cpu': -2.0, 'test_memory': -2.0, 'test_disk': -2.0,
    'cpu_percent': -1.5, 'virtual_memory': -1.5, 'disk_io': -1.5,

    # Documentation/metadata patterns
    'copyright': -0.5, 'license': -0.5, 'warranty': -0.8,
    'autogenerated': -1.5, 'auto-generated': -1.5,
    'do not edit': -2.0, 'generated by': -1.0,
    'permission is hereby granted': -2.0,

    # Clearly unrelated applications
    'recipe': -3.0, 'cooking': -3.0, 'ingredient': -3.0,
    'game': -2.5, 'pygame': -3.5, 'game_engine': -3.0, 'game loop': -3.0,
    'sprite': -2.0, 'pygame.init': -3.5, 'pygame.display': -3.0,
    'django': -2.5, 'flask': -2.0, 'fastapi': -1.5,
    'httpresponse': -2.0, 'render_template': -2.0,
}

# Path patterns that indicate relevance
POSITIVE_PATH_PATTERNS = {
    'src/options': 2.5, 'src/broker': 2.5, 'src/trading': 2.5,
    'src/news': 1.5, 'src/utils': 0.5,
    'monitoring': 1.5, 'ml/': 1.5, 'config/': 1.0,
    'tools/ibg': 2.0, 'ops/ibg': 2.0,
}

# Path patterns that indicate irrelevance
NEGATIVE_PATH_PATTERNS = {
    'site-packages': -5.0, 'dist-info': -5.0, '.wfo_venv': -5.0,
    'venv/': -5.0, 'node_modules': -5.0,
    'tests/test_': -1.0,  # Be careful - our tests might be relevant
    '.egg-info': -3.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeywordScore:
    """Result of keyword-based pre-analysis."""
    score: float
    positive_matches: list[str]
    negative_matches: list[str]
    path_score: float

    @property
    def total_score(self) -> float:
        return self.score + self.path_score

    @property
    def is_clearly_relevant(self) -> bool:
        """Score high enough to skip AI analysis."""
        # High score with no negatives = definitely relevant
        if self.total_score >= 5.0 and not self.negative_matches:
            return True
        # Very high score even with some negatives = still relevant
        if self.total_score >= 12.0:
            return True
        return False

    @property
    def is_clearly_irrelevant(self) -> bool:
        """Score low enough to skip AI analysis."""
        # Very negative score = definitely not relevant
        if self.total_score <= -5.0:
            return True
        # Path indicates third-party with weak content score
        if self.path_score <= -4.0 and self.score < 2.0:
            return True
        # Multiple strong negative keyword matches
        if len(self.negative_matches) >= 2 and self.total_score <= -3.0:
            return True
        return False

    @property
    def hint_text(self) -> str:
        """Generate hint for AI prompt."""
        if self.total_score > 5:
            return "LIKELY RELEVANT (strong trading keywords found)"
        elif self.total_score > 2:
            return "POSSIBLY RELEVANT (some trading keywords)"
        elif self.total_score < -3:
            return "LIKELY NOT RELEVANT (third-party library patterns)"
        elif self.total_score < 0:
            return "POSSIBLY NOT RELEVANT (few trading indicators)"
        else:
            return "UNCERTAIN (mixed signals)"


def compute_keyword_score(content: str, filepath: str) -> KeywordScore:
    """
    Compute relevance score based on keyword matching.

    This is a fast pre-filter before calling the LLM.
    """
    content_lower = content.lower()
    filepath_lower = filepath.lower()

    score = 0.0
    positive_matches = []
    negative_matches = []

    # Check strong positive keywords
    for keyword, weight in STRONG_POSITIVE_KEYWORDS.items():
        if keyword.lower() in content_lower:
            score += weight
            positive_matches.append(keyword)

    # Check moderate positive keywords
    for keyword, weight in MODERATE_POSITIVE_KEYWORDS.items():
        if keyword.lower() in content_lower:
            score += weight
            if weight >= 1.0:
                positive_matches.append(keyword)

    # Check strong negative keywords
    for keyword, weight in STRONG_NEGATIVE_KEYWORDS.items():
        if keyword.lower() in content_lower:
            score += weight  # weight is already negative
            negative_matches.append(keyword)

    # Path-based scoring
    path_score = 0.0
    for pattern, weight in POSITIVE_PATH_PATTERNS.items():
        if pattern in filepath_lower:
            path_score += weight

    for pattern, weight in NEGATIVE_PATH_PATTERNS.items():
        if pattern in filepath_lower:
            path_score += weight  # weight is already negative

    return KeywordScore(
        score=score,
        positive_matches=positive_matches[:5],  # Limit for display
        negative_matches=negative_matches[:5],
        path_score=path_score,
    )


def calibrate_confidence(
    ai_confidence: float,
    keyword_score: KeywordScore,
    ai_relevant: bool,
) -> float:
    """
    Calibrate AI confidence based on keyword analysis.

    Adjusts confidence when AI and keyword analysis agree/disagree.
    More aggressive when keywords strongly disagree with AI.
    """
    total_kw_score = keyword_score.total_score
    has_strong_negatives = len(keyword_score.negative_matches) >= 2

    # AI says relevant
    if ai_relevant:
        if total_kw_score >= 5:
            # Strong keyword support - boost confidence
            return min(ai_confidence + 0.1, 0.99)
        elif total_kw_score <= -5 or has_strong_negatives:
            # Strong negative keywords - heavily penalize (likely false positive)
            return max(ai_confidence - 0.5, 0.25)
        elif total_kw_score <= -2:
            # Moderate negative keywords - significant reduction
            return max(ai_confidence - 0.35, 0.3)
        elif total_kw_score <= 0:
            # Weak keyword support - reduction
            return max(ai_confidence - 0.2, 0.35)
        else:
            return ai_confidence

    # AI says not relevant
    else:
        if total_kw_score <= -3:
            # Strong keyword support - boost confidence
            return min(ai_confidence + 0.15, 0.99)
        elif total_kw_score >= 8:
            # Very strong positive keywords - heavily penalize AI decision
            return max(ai_confidence - 0.5, 0.25)
        elif total_kw_score >= 5:
            # Strong positive keywords - significant reduction
            return max(ai_confidence - 0.35, 0.3)
        elif total_kw_score >= 2:
            # Some positive keywords - moderate reduction
            return max(ai_confidence - 0.2, 0.35)
        else:
            return ai_confidence


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class Relevance(Enum):
    """File relevance classification."""
    RELEVANT = "relevant"           # Should be included in repo
    NOT_RELEVANT = "not_relevant"   # Should be excluded
    UNCERTAIN = "uncertain"         # Needs manual review
    ERROR = "error"                 # Analysis failed


@dataclass
class FileAnalysis:
    """Result of analyzing a single file."""
    path: str
    relevance: Relevance
    confidence: float  # 0.0 to 1.0
    reason: str
    content_hash: str
    analyzed_at: str
    model_used: str = DEFAULT_MODEL

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "relevance": self.relevance.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "content_hash": self.content_hash,
            "analyzed_at": self.analyzed_at,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileAnalysis":
        return cls(
            path=data["path"],
            relevance=Relevance(data["relevance"]),
            confidence=data["confidence"],
            reason=data["reason"],
            content_hash=data["content_hash"],
            analyzed_at=data["analyzed_at"],
            model_used=data.get("model_used", DEFAULT_MODEL),
        )


@dataclass
class BaselineCache:
    """Persistent cache of file analyses."""
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    files: dict[str, FileAnalysis] = field(default_factory=dict)

    def save(self, path: Path = BASELINE_FILE):
        """Save cache to disk."""
        self.updated_at = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = self.updated_at

        data = {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = BASELINE_FILE) -> "BaselineCache":
        """Load cache from disk."""
        if not path.exists():
            return cls()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cache = cls(
                version=data.get("version", "1.0"),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
            )
            for filepath, analysis_data in data.get("files", {}).items():
                cache.files[filepath] = FileAnalysis.from_dict(analysis_data)
            return cache
        except Exception as e:
            print(f"Warning: Could not load baseline cache: {e}")
            return cls()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Simple client for Ollama API with GPU-efficient model management.

    Uses REST API with keep_alive=0 to immediately unload the model from GPU
    memory after each request, preventing continuous GPU usage.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        keep_alive: int = 0,  # Seconds to keep model loaded (0 = unload immediately)
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self._session = None

    def _get_session(self):
        """Lazy-load requests session."""
        if self._session is None:
            import urllib.request
        return urllib.request

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                # Check if our model (or base model name) is available
                model_base = self.model.split(":")[0]
                return any(model_base in m for m in models)
        except Exception:
            # Fallback to CLI check
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return self.model.split(":")[0] in result.stdout
            except Exception:
                return False

    def generate(self, prompt: str, timeout: int = 30) -> str:
        """
        Generate a response from the model using REST API.

        Uses keep_alive="0" to immediately unload model from GPU after response,
        preventing continuous GPU memory usage when idle.
        """
        try:
            import urllib.request
            import urllib.error

            # Ollama expects keep_alive as string: "0" for immediate unload, "5m" for 5 minutes
            keep_alive_str = str(self.keep_alive) if self.keep_alive == 0 else f"{self.keep_alive}s"

            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive_str,  # "0" = unload immediately after response
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode())
                return data.get("response", "").strip()

        except urllib.error.URLError as e:
            return f"ERROR: Ollama connection failed: {e}"
        except TimeoutError:
            return "ERROR: Model timeout"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def unload_model(self) -> bool:
        """
        Explicitly unload the model from GPU memory.

        This can be called to ensure the model is unloaded even if keep_alive > 0.
        """
        try:
            import urllib.request

            # Setting keep_alive to "0" with an empty prompt triggers unload
            payload = json.dumps({
                "model": self.model,
                "prompt": "",
                "keep_alive": "0",  # Must be string "0" for immediate unload
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# AI Relevance Detector
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are classifying files for an SPY vertical spread options trading system.

RELEVANT files directly relate to:
- SPY options trading (vertical spreads, calls, puts, strikes, expiration)
- Interactive Brokers integration (ib_insync, TWS, IB Gateway)
- Options Greeks (delta, gamma, theta, vega, Black-Scholes)
- Market data feeds (Databento, OPRA, Polygon)
- Trading monitoring, execution, risk management
- ML models for trade signals

NOT RELEVANT files are:
- Third-party library internals (matplotlib, psutil, PIL tests)
- Generic utilities not specific to trading
- Documentation/licenses/metadata files
- Unrelated applications (games, web frameworks, recipes)

EXAMPLES:

File: options_executor.py
Content: "from ib_insync import IB; def execute_vertical_spread(strike_long, strike_short)..."
Answer: {{"relevant": true, "confidence": 0.95, "reason": "IBKR vertical spread execution"}}

File: test_art3d.py
Content: "import matplotlib; def test_3d_plotting()..."
Answer: {{"relevant": false, "confidence": 0.95, "reason": "matplotlib library test, not trading"}}

File: greeks_calculator.py
Content: "def black_scholes_delta(S, K, T, r, sigma)..."
Answer: {{"relevant": true, "confidence": 0.90, "reason": "Black-Scholes Greeks for options"}}

File: six.py
Content: "Python 2 and 3 compatibility utilities..."
Answer: {{"relevant": false, "confidence": 0.98, "reason": "Generic Python compatibility library"}}

NOW CLASSIFY THIS FILE:
File: {filename}
Path hint: {path_hint}
Keyword score: {keyword_score}
Content:
```
{content}
```

Respond with ONLY a JSON object:"""


class AIRelevanceDetector:
    """
    AI-powered file relevance detector using local Ollama models.

    Features:
    - Lazy initialization: Ollama is only contacted when actually analyzing files
    - GPU-efficient: Uses keep_alive=0 to unload model immediately after each request
    - Two-pass analysis: Keyword pre-scoring handles obvious cases without AI

    Usage:
        detector = AIRelevanceDetector()
        result = detector.analyze_file("/path/to/file.py")
        if result.relevance == Relevance.RELEVANT:
            print("File should be included")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_path: Path = BASELINE_FILE,
        use_cache: bool = True,
        keep_alive: int = 0,  # GPU unload delay (0 = immediate)
    ):
        self._model = model
        self._keep_alive = keep_alive
        self._client: OllamaClient | None = None  # Lazy initialized
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.cache = BaselineCache.load(cache_path) if use_cache else BaselineCache()

    @property
    def client(self) -> OllamaClient:
        """Lazy-initialize the Ollama client only when needed."""
        if self._client is None:
            self._client = OllamaClient(
                model=self._model,
                keep_alive=self._keep_alive,
            )
        return self._client

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _should_skip_file(self, filepath: Path) -> bool:
        """Check if file should be skipped."""
        # Skip by extension
        if filepath.suffix.lower() not in ANALYZABLE_EXTENSIONS:
            return True

        # Skip by filename
        if filepath.name in SKIP_FILES:
            return True

        # Skip by directory
        for part in filepath.parts:
            if part in SKIP_DIRECTORIES:
                return True

        return False

    def _read_file_content(self, filepath: Path) -> str | None:
        """Read file content, return None if unreadable."""
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            # Truncate for model context limit
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n... [truncated]"
            return content
        except Exception:
            return None

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse model response into structured data."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            # Find JSON object
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                relevant = data.get("relevant", False)
                confidence = float(data.get("confidence", 0.5))
                reason = data.get("reason", "No reason provided")

                return relevant, confidence, reason
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: try to interpret YES/NO
        response_lower = response.lower()
        if "yes" in response_lower or "true" in response_lower or "relevant" in response_lower:
            return True, 0.6, "Interpreted as relevant from response"
        elif "no" in response_lower or "false" in response_lower:
            return False, 0.6, "Interpreted as not relevant from response"

        return False, 0.3, f"Could not parse response: {response[:100]}"

    def analyze_file(self, filepath: str | Path, force: bool = False) -> FileAnalysis:
        """
        Analyze a single file for relevance using two-pass analysis.

        Pass 1: Fast keyword scoring to catch obvious cases
        Pass 2: AI analysis for ambiguous files (with keyword hints)

        Args:
            filepath: Path to the file
            force: If True, ignore cache and re-analyze

        Returns:
            FileAnalysis with relevance determination
        """
        filepath = Path(filepath).resolve()
        path_key = str(filepath)

        # Check if should skip
        if self._should_skip_file(filepath):
            return FileAnalysis(
                path=path_key,
                relevance=Relevance.NOT_RELEVANT,
                confidence=1.0,
                reason="Skipped by file type/name rules",
                content_hash="",
                analyzed_at=datetime.utcnow().isoformat(),
            )

        # Read content
        content = self._read_file_content(filepath)
        if content is None:
            return FileAnalysis(
                path=path_key,
                relevance=Relevance.ERROR,
                confidence=0.0,
                reason="Could not read file",
                content_hash="",
                analyzed_at=datetime.utcnow().isoformat(),
            )

        content_hash = self._compute_hash(content)

        # Check cache
        if self.use_cache and not force and path_key in self.cache.files:
            cached = self.cache.files[path_key]
            if cached.content_hash == content_hash:
                return cached

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 1: Keyword-based pre-scoring
        # ═══════════════════════════════════════════════════════════════════════
        kw_score = compute_keyword_score(content, path_key)

        # Fast path: clearly irrelevant (third-party library, venv, etc.)
        if kw_score.is_clearly_irrelevant:
            neg_keywords = ", ".join(kw_score.negative_matches[:3]) if kw_score.negative_matches else "path pattern"
            analysis = FileAnalysis(
                path=path_key,
                relevance=Relevance.NOT_RELEVANT,
                confidence=0.95,
                reason=f"Keyword filter: {neg_keywords} (score: {kw_score.total_score:.1f})",
                content_hash=content_hash,
                analyzed_at=datetime.utcnow().isoformat(),
                model_used="keyword_filter",
            )
            if self.use_cache:
                self.cache.files[path_key] = analysis
                self.cache.save(self.cache_path)
            return analysis

        # Fast path: clearly relevant (strong trading keywords, no negatives)
        if kw_score.is_clearly_relevant:
            pos_keywords = ", ".join(kw_score.positive_matches[:3])
            analysis = FileAnalysis(
                path=path_key,
                relevance=Relevance.RELEVANT,
                confidence=0.90,
                reason=f"Keyword filter: {pos_keywords} (score: {kw_score.total_score:.1f})",
                content_hash=content_hash,
                analyzed_at=datetime.utcnow().isoformat(),
                model_used="keyword_filter",
            )
            if self.use_cache:
                self.cache.files[path_key] = analysis
                self.cache.save(self.cache_path)
            return analysis

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 2: AI analysis (with keyword hints)
        # ═══════════════════════════════════════════════════════════════════════
        prompt = ANALYSIS_PROMPT.format(
            filename=filepath.name,
            path_hint=kw_score.hint_text,
            keyword_score=f"{kw_score.total_score:+.1f} (positive: {kw_score.positive_matches[:3]}, negative: {kw_score.negative_matches[:3]})",
            content=content,
        )

        response = self.client.generate(prompt)

        if response.startswith("ERROR:"):
            return FileAnalysis(
                path=path_key,
                relevance=Relevance.ERROR,
                confidence=0.0,
                reason=response,
                content_hash=content_hash,
                analyzed_at=datetime.utcnow().isoformat(),
            )

        # Parse response
        relevant, raw_confidence, reason = self._parse_response(response)

        # ═══════════════════════════════════════════════════════════════════════
        # Confidence calibration and decision override based on keyword agreement
        # ═══════════════════════════════════════════════════════════════════════
        confidence = calibrate_confidence(raw_confidence, kw_score, relevant)

        # Override AI decision if keywords strongly disagree and AI confidence was low
        override_reason = None
        if relevant and kw_score.total_score <= -4 and raw_confidence < 0.8:
            # AI says relevant but keywords strongly disagree - flip decision
            relevant = False
            confidence = 0.7
            override_reason = f"Overridden: keywords strongly negative (score={kw_score.total_score:.1f}, negatives={kw_score.negative_matches})"
        elif not relevant and kw_score.total_score >= 6 and raw_confidence < 0.8:
            # AI says not relevant but keywords strongly disagree - flip decision
            relevant = True
            confidence = 0.7
            override_reason = f"Overridden: keywords strongly positive (score={kw_score.total_score:.1f}, positives={kw_score.positive_matches})"

        # Add context to reason
        if override_reason:
            reason = override_reason
        elif relevant and kw_score.total_score < -2:
            reason = f"{reason} [WARN: keywords suggest not relevant, score={kw_score.total_score:.1f}]"
        elif not relevant and kw_score.total_score > 3:
            reason = f"{reason} [WARN: keywords suggest relevant, score={kw_score.total_score:.1f}]"

        # Determine final relevance
        if confidence < 0.4:
            relevance = Relevance.UNCERTAIN
        elif relevant:
            relevance = Relevance.RELEVANT
        else:
            relevance = Relevance.NOT_RELEVANT

        analysis = FileAnalysis(
            path=path_key,
            relevance=relevance,
            confidence=confidence,
            reason=reason,
            content_hash=content_hash,
            analyzed_at=datetime.utcnow().isoformat(),
            model_used=self.client.model,
        )

        # Update cache
        if self.use_cache:
            self.cache.files[path_key] = analysis
            self.cache.save(self.cache_path)

        return analysis

    def analyze_directory(
        self,
        directory: str | Path,
        force: bool = False,
        verbose: bool = False,
    ) -> Iterator[FileAnalysis]:
        """
        Analyze all files in a directory.

        Args:
            directory: Root directory to scan
            force: Re-analyze even if cached
            verbose: Print progress

        Yields:
            FileAnalysis for each file
        """
        directory = Path(directory).resolve()

        for root, dirs, files in os.walk(directory):
            # Skip excluded directories (exact match or pattern-based)
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRECTORIES
                and not any(pattern in d for pattern in SKIP_DIR_PATTERNS)
            ]

            for filename in files:
                filepath = Path(root) / filename

                if self._should_skip_file(filepath):
                    continue

                if verbose:
                    print(f"Analyzing: {filepath.relative_to(directory)}", end=" ... ")

                try:
                    result = self.analyze_file(filepath, force=force)
                    if verbose:
                        status = "✓" if result.relevance == Relevance.RELEVANT else "✗"
                        print(f"{status} ({result.confidence:.0%})")
                    yield result
                except Exception as e:
                    if verbose:
                        print(f"ERROR: {e}")

    def get_relevant_files(self) -> list[str]:
        """Get list of files marked as relevant in cache."""
        return [
            path for path, analysis in self.cache.files.items()
            if analysis.relevance == Relevance.RELEVANT
        ]

    def get_statistics(self) -> dict:
        """Get statistics about cached analyses."""
        stats = {
            "total": len(self.cache.files),
            "relevant": 0,
            "not_relevant": 0,
            "uncertain": 0,
            "error": 0,
        }
        for analysis in self.cache.files.values():
            stats[analysis.relevance.value] = stats.get(analysis.relevance.value, 0) + 1
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def check_file_relevance(filepath: str | Path) -> bool:
    """Quick check if a file is relevant to the trading system."""
    detector = AIRelevanceDetector()
    result = detector.analyze_file(filepath)
    return result.relevance == Relevance.RELEVANT


def run_baseline_scan(
    directories: list[str | Path],
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run a full baseline scan of directories.

    Returns dict with scan results and statistics.
    """
    detector = AIRelevanceDetector()

    if not detector.client.is_available():
        raise RuntimeError("Ollama is not available. Please start Ollama first.")

    results = {
        "relevant": [],
        "not_relevant": [],
        "uncertain": [],
        "errors": [],
    }

    for directory in directories:
        directory = Path(directory).expanduser().resolve()
        if not directory.exists():
            print(f"Warning: Directory does not exist: {directory}")
            continue

        if verbose:
            print(f"\nScanning: {directory}")
            print("=" * 60)

        for analysis in detector.analyze_directory(directory, force=force, verbose=verbose):
            if analysis.relevance == Relevance.RELEVANT:
                results["relevant"].append(analysis.path)
            elif analysis.relevance == Relevance.NOT_RELEVANT:
                results["not_relevant"].append(analysis.path)
            elif analysis.relevance == Relevance.UNCERTAIN:
                results["uncertain"].append(analysis.path)
            else:
                results["errors"].append(analysis.path)

    results["statistics"] = detector.get_statistics()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ai_relevance_detector.py <file_or_directory>")
        print("       python ai_relevance_detector.py --status")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--status":
        detector = AIRelevanceDetector()
        print("AI Relevance Detector Status")
        print("=" * 40)
        print(f"Model: {detector.client.model}")
        print(f"Ollama available: {detector.client.is_available()}")
        print(f"Cache file: {detector.cache_path}")
        print(f"Cached files: {len(detector.cache.files)}")
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    path = Path(arg)
    detector = AIRelevanceDetector()

    if not detector.client.is_available():
        print("ERROR: Ollama is not available. Please start Ollama first.")
        sys.exit(1)

    if path.is_file():
        result = detector.analyze_file(path)
        print(f"\nFile: {path}")
        print(f"Relevance: {result.relevance.value}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Reason: {result.reason}")
    elif path.is_dir():
        print(f"Scanning directory: {path}")
        for result in detector.analyze_directory(path, verbose=True):
            pass
        print(f"\nStatistics: {detector.get_statistics()}")
    else:
        print(f"ERROR: Path does not exist: {path}")
        sys.exit(1)
