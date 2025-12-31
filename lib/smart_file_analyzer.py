#!/usr/bin/env python3
"""
smart_file_analyzer.py - Intelligent File Dependency and Relevance Analyzer

A production-grade file analysis system that uses AST parsing, dependency graphs,
and keyword analysis to intelligently determine which files are related to the
SPY vertical spread trading system.

Features:
- Python AST import analysis for dependency tracking
- Dependency graph construction using NetworkX
- Transitive dependency resolution
- Keyword-based relevance scoring
- Config file detection (YAML, TOML, JSON)
- Automatic exclusion recommendations
- File categorization (core, support, config, data, unrelated)

References:
- Python AST: https://docs.python.org/3/library/ast.html
- NetworkX: https://networkx.org/
- Snakefood approach: https://furius.ca/snakefood/

Author: Clone System v2.0
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterator

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


class FileCategory(Enum):
    """Categories for file relevance classification."""
    CORE = auto()           # Core trading system files
    SUPPORT = auto()        # Supporting utilities/helpers
    CONFIG = auto()         # Configuration files
    DATA = auto()           # Data files (should often be excluded)
    TEST = auto()           # Test files
    DOCUMENTATION = auto()  # Documentation
    BUILD = auto()          # Build/deployment files
    UNRELATED = auto()      # Not related to trading system
    UNKNOWN = auto()        # Cannot determine


@dataclass
class FileInfo:
    """Information about an analyzed file."""
    path: Path
    category: FileCategory
    relevance_score: float  # 0.0 to 1.0
    imports: set[str] = field(default_factory=set)
    imported_by: set[str] = field(default_factory=set)
    keywords_found: list[str] = field(default_factory=list)
    size_bytes: int = 0
    is_python: bool = False
    should_include: bool = True
    reason: str = ""


@dataclass
class AnalysisResult:
    """Result of codebase analysis."""
    files: dict[str, FileInfo]
    dependency_graph: dict[str, set[str]]
    core_modules: set[str]
    recommended_includes: set[str]
    recommended_excludes: set[str]
    statistics: dict[str, int]


# ─────────────────────────────────────────────────────────────────────────────
# Trading System Keywords
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that indicate trading system relevance (weighted)
TRADING_KEYWORDS = {
    # High relevance (weight 1.0)
    'spy': 1.0,
    'vertical_spread': 1.0,
    'vertical spread': 1.0,
    'options': 0.9,
    'option': 0.9,
    'spread': 0.8,
    'trading': 0.9,
    'trade': 0.8,
    'strategy': 0.7,
    'position': 0.7,
    'order': 0.7,
    'execution': 0.7,

    # Broker/Data related (weight 0.8)
    'ibkr': 0.9,
    'interactive_brokers': 0.9,
    'interactive brokers': 0.9,
    'tws': 0.8,
    'ib_insync': 0.9,
    'databento': 0.8,
    'polygon': 0.7,
    'opra': 0.8,
    'cbbo': 0.8,

    # Options specific (weight 0.8)
    'strike': 0.8,
    'expiration': 0.8,
    'expiry': 0.8,
    'call': 0.6,
    'put': 0.6,
    'greeks': 0.8,
    'delta': 0.7,
    'gamma': 0.7,
    'theta': 0.7,
    'vega': 0.7,
    'iv': 0.6,
    'implied_volatility': 0.8,
    'black_scholes': 0.8,

    # ML/Model related (weight 0.7)
    'model': 0.5,
    'predict': 0.6,
    'feature': 0.5,
    'training': 0.6,
    'backtest': 0.8,
    'wfo': 0.8,
    'walk_forward': 0.8,
    'walk forward': 0.8,

    # Data related (weight 0.6)
    'ohlc': 0.7,
    'candlestick': 0.6,
    'market_data': 0.7,
    'quote': 0.6,
    'ticker': 0.6,
    'symbol': 0.5,
    'underlier': 0.7,
    'underlying': 0.7,

    # Risk/Portfolio (weight 0.7)
    'risk': 0.6,
    'portfolio': 0.7,
    'pnl': 0.8,
    'profit': 0.6,
    'loss': 0.5,
    'margin': 0.7,

    # Infrastructure (weight 0.5)
    'ingest': 0.6,
    'pipeline': 0.5,
    'scheduler': 0.5,
    'monitor': 0.5,
    'metrics': 0.5,
    'telemetry': 0.5,
}

# Keywords that indicate file should be excluded
EXCLUDE_KEYWORDS = {
    'test_',
    '_test.py',
    'tests/',
    '__pycache__',
    '.pyc',
    '.pyo',
    'node_modules',
    '.git',
    '.venv',
    'venv/',
    'env/',
    '.egg-info',
    'dist/',
    'build/',
    '.tox',
    '.pytest_cache',
    '.mypy_cache',
    '.ipynb_checkpoints',
}

# File extensions to analyze
PYTHON_EXTENSIONS = {'.py', '.pyx', '.pyi'}
CONFIG_EXTENSIONS = {'.yaml', '.yml', '.toml', '.json', '.ini', '.cfg', '.conf'}
DATA_EXTENSIONS = {'.csv', '.parquet', '.feather', '.pkl', '.pickle', '.h5', '.hdf5'}
DOC_EXTENSIONS = {'.md', '.rst', '.txt'}


# ─────────────────────────────────────────────────────────────────────────────
# AST Import Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements."""

    def __init__(self):
        self.imports: set[str] = set()
        self.from_imports: dict[str, set[str]] = defaultdict(set)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split('.')[0]
            self.imports.add(base_module)
            for alias in node.names:
                self.from_imports[node.module].add(alias.name)
        self.generic_visit(node)


def extract_imports(filepath: Path) -> set[str]:
    """Extract imports from a Python file using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except (SyntaxError, UnicodeDecodeError, Exception):
        return set()


def extract_imports_with_details(filepath: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Extract detailed import information from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports, dict(visitor.from_imports)
    except (SyntaxError, UnicodeDecodeError, Exception):
        return set(), {}


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def analyze_keywords(content: str, filepath: Path) -> tuple[float, list[str]]:
    """
    Analyze content for trading-related keywords.

    Returns:
        Tuple of (relevance_score, list of found keywords)
    """
    content_lower = content.lower()
    filename_lower = str(filepath).lower()

    found_keywords = []
    total_weight = 0.0
    max_possible = sum(TRADING_KEYWORDS.values())

    for keyword, weight in TRADING_KEYWORDS.items():
        # Check in content
        if keyword in content_lower:
            found_keywords.append(keyword)
            total_weight += weight
        # Check in filename (higher weight)
        elif keyword.replace(' ', '_') in filename_lower or keyword.replace('_', '') in filename_lower:
            found_keywords.append(f"{keyword} (filename)")
            total_weight += weight * 1.5

    # Normalize score
    relevance_score = min(total_weight / (max_possible * 0.3), 1.0)  # 30% of max = 1.0

    return relevance_score, found_keywords


def should_exclude_path(filepath: Path) -> tuple[bool, str]:
    """Check if a path should be excluded based on patterns."""
    path_str = str(filepath).lower()

    for pattern in EXCLUDE_KEYWORDS:
        if pattern in path_str:
            return True, f"Matches exclude pattern: {pattern}"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# File Categorizer
# ─────────────────────────────────────────────────────────────────────────────

def categorize_file(filepath: Path, relevance_score: float, keywords: list[str]) -> FileCategory:
    """Categorize a file based on its path, extension, and content analysis."""
    path_str = str(filepath).lower()
    suffix = filepath.suffix.lower()

    # Test files
    if 'test' in path_str or suffix == '.test.py':
        return FileCategory.TEST

    # Documentation
    if suffix in DOC_EXTENSIONS or 'readme' in path_str or 'doc' in path_str:
        return FileCategory.DOCUMENTATION

    # Data files
    if suffix in DATA_EXTENSIONS:
        return FileCategory.DATA

    # Config files
    if suffix in CONFIG_EXTENSIONS or 'config' in path_str or 'settings' in path_str:
        return FileCategory.CONFIG

    # Build files
    if any(p in path_str for p in ['setup.py', 'pyproject.toml', 'makefile', 'dockerfile']):
        return FileCategory.BUILD

    # Python files - categorize by relevance
    if suffix in PYTHON_EXTENSIONS:
        if relevance_score >= 0.6:
            return FileCategory.CORE
        elif relevance_score >= 0.3:
            return FileCategory.SUPPORT
        elif relevance_score > 0:
            return FileCategory.SUPPORT
        else:
            return FileCategory.UNRELATED

    return FileCategory.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Smart File Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SmartFileAnalyzer:
    """
    Intelligent file analyzer for trading system codebases.

    Uses AST parsing, dependency graphs, and keyword analysis to determine
    which files are related to the trading system and should be included
    in the clone.
    """

    def __init__(
        self,
        root_path: Path,
        entry_points: list[str] | None = None,
        additional_keywords: dict[str, float] | None = None,
        min_relevance: float = 0.2,
    ):
        """
        Initialize the analyzer.

        Args:
            root_path: Root directory to analyze
            entry_points: Known entry point modules (e.g., ['cli', 'main'])
            additional_keywords: Additional domain-specific keywords
            min_relevance: Minimum relevance score to include
        """
        self.root_path = Path(root_path).resolve()
        self.entry_points = set(entry_points or [])
        self.min_relevance = min_relevance

        # Merge additional keywords
        self.keywords = dict(TRADING_KEYWORDS)
        if additional_keywords:
            self.keywords.update(additional_keywords)

        # Analysis results
        self.files: dict[str, FileInfo] = {}
        self.dependency_graph: dict[str, set[str]] = defaultdict(set)
        self.reverse_graph: dict[str, set[str]] = defaultdict(set)

    def _iter_python_files(self) -> Iterator[Path]:
        """Iterate over Python files in the codebase."""
        for root, dirs, files in os.walk(self.root_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not should_exclude_path(Path(root) / d)[0]]

            for filename in files:
                filepath = Path(root) / filename
                if filepath.suffix in PYTHON_EXTENSIONS:
                    yield filepath

    def _iter_all_files(self) -> Iterator[Path]:
        """Iterate over all relevant files in the codebase."""
        for root, dirs, files in os.walk(self.root_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not should_exclude_path(Path(root) / d)[0]]

            for filename in files:
                filepath = Path(root) / filename
                excluded, _ = should_exclude_path(filepath)
                if not excluded:
                    yield filepath

    def _resolve_module_path(self, module_name: str) -> Path | None:
        """Try to resolve a module name to a file path."""
        # Try direct file
        for ext in PYTHON_EXTENSIONS:
            candidate = self.root_path / f"{module_name.replace('.', '/')}{ext}"
            if candidate.exists():
                return candidate

        # Try package (directory with __init__.py)
        candidate = self.root_path / module_name.replace('.', '/') / '__init__.py'
        if candidate.exists():
            return candidate

        return None

    def analyze(self) -> AnalysisResult:
        """
        Perform full analysis of the codebase.

        Returns:
            AnalysisResult with all analysis data
        """
        # Phase 1: Scan all Python files and extract imports
        for filepath in self._iter_python_files():
            self._analyze_python_file(filepath)

        # Phase 2: Scan config and other files
        for filepath in self._iter_all_files():
            if filepath.suffix not in PYTHON_EXTENSIONS:
                self._analyze_other_file(filepath)

        # Phase 3: Build dependency graph
        self._build_dependency_graph()

        # Phase 4: Identify core modules (entry points + high relevance)
        core_modules = self._identify_core_modules()

        # Phase 5: Propagate relevance through dependencies
        self._propagate_relevance(core_modules)

        # Phase 6: Generate recommendations
        recommended_includes, recommended_excludes = self._generate_recommendations()

        # Compile statistics
        statistics = self._compile_statistics()

        return AnalysisResult(
            files=self.files,
            dependency_graph=dict(self.dependency_graph),
            core_modules=core_modules,
            recommended_includes=recommended_includes,
            recommended_excludes=recommended_excludes,
            statistics=statistics,
        )

    def _analyze_python_file(self, filepath: Path):
        """Analyze a Python file."""
        rel_path = filepath.relative_to(self.root_path)
        key = str(rel_path)

        # Read content
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            content = ""

        # Extract imports
        imports = extract_imports(filepath)

        # Analyze keywords
        relevance_score, keywords_found = analyze_keywords(content, filepath)

        # Check exclusion
        excluded, reason = should_exclude_path(filepath)

        # Categorize
        category = categorize_file(filepath, relevance_score, keywords_found)

        self.files[key] = FileInfo(
            path=filepath,
            category=category,
            relevance_score=relevance_score,
            imports=imports,
            keywords_found=keywords_found,
            size_bytes=filepath.stat().st_size if filepath.exists() else 0,
            is_python=True,
            should_include=not excluded and relevance_score >= self.min_relevance,
            reason=reason if excluded else "",
        )

        # Add to dependency graph
        self.dependency_graph[key] = imports

    def _analyze_other_file(self, filepath: Path):
        """Analyze a non-Python file."""
        rel_path = filepath.relative_to(self.root_path)
        key = str(rel_path)

        if key in self.files:
            return

        # Read content for keyword analysis
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            content = ""

        # Analyze keywords
        relevance_score, keywords_found = analyze_keywords(content, filepath)

        # Check exclusion
        excluded, reason = should_exclude_path(filepath)

        # Categorize
        category = categorize_file(filepath, relevance_score, keywords_found)

        # Config files related to trading should be included
        if category == FileCategory.CONFIG and relevance_score > 0:
            relevance_score = max(relevance_score, 0.5)

        self.files[key] = FileInfo(
            path=filepath,
            category=category,
            relevance_score=relevance_score,
            keywords_found=keywords_found,
            size_bytes=filepath.stat().st_size if filepath.exists() else 0,
            is_python=False,
            should_include=not excluded and (
                relevance_score >= self.min_relevance or
                category in (FileCategory.CONFIG, FileCategory.BUILD)
            ),
            reason=reason if excluded else "",
        )

    def _build_dependency_graph(self):
        """Build the reverse dependency graph."""
        for file_key, file_info in self.files.items():
            if not file_info.is_python:
                continue

            for imported_module in file_info.imports:
                # Try to find the imported module in our codebase
                for other_key, other_info in self.files.items():
                    if not other_info.is_python:
                        continue

                    # Check if the module name matches
                    module_name = Path(other_key).stem
                    if module_name == imported_module or other_key.startswith(f"{imported_module}/"):
                        self.reverse_graph[other_key].add(file_key)
                        file_info.imported_by.add(other_key)

    def _identify_core_modules(self) -> set[str]:
        """Identify core modules based on entry points and relevance."""
        core = set()

        for file_key, file_info in self.files.items():
            # High relevance files are core
            if file_info.relevance_score >= 0.6:
                core.add(file_key)
                continue

            # Entry points are core
            module_name = Path(file_key).stem
            if module_name in self.entry_points:
                core.add(file_key)
                continue

            # Files with many trading keywords are core
            if len(file_info.keywords_found) >= 5:
                core.add(file_key)

        return core

    def _propagate_relevance(self, core_modules: set[str]):
        """Propagate relevance through dependency graph."""
        if not HAS_NETWORKX:
            return

        # Build NetworkX graph
        G = nx.DiGraph()
        for file_key, imports in self.dependency_graph.items():
            G.add_node(file_key)
            for imp in imports:
                # Find matching files
                for other_key in self.files:
                    if Path(other_key).stem == imp:
                        G.add_edge(file_key, other_key)

        # For each core module, mark its dependencies as relevant
        for core in core_modules:
            if core not in G:
                continue

            # Get all reachable nodes (dependencies)
            try:
                reachable = nx.descendants(G, core)
                for dep in reachable:
                    if dep in self.files:
                        # Boost relevance of dependencies
                        current = self.files[dep].relevance_score
                        self.files[dep].relevance_score = min(current + 0.2, 1.0)
                        if self.files[dep].relevance_score >= self.min_relevance:
                            self.files[dep].should_include = True
                            self.files[dep].reason = f"Dependency of core module"
            except nx.NetworkXError:
                pass

    def _generate_recommendations(self) -> tuple[set[str], set[str]]:
        """Generate include/exclude recommendations."""
        includes = set()
        excludes = set()

        for file_key, file_info in self.files.items():
            if file_info.should_include:
                includes.add(file_key)
            else:
                if file_info.category in (FileCategory.DATA, FileCategory.TEST):
                    excludes.add(file_key)
                elif file_info.relevance_score < 0.1:
                    excludes.add(file_key)

        return includes, excludes

    def _compile_statistics(self) -> dict[str, int]:
        """Compile analysis statistics."""
        stats = {
            'total_files': len(self.files),
            'python_files': sum(1 for f in self.files.values() if f.is_python),
            'included_files': sum(1 for f in self.files.values() if f.should_include),
            'excluded_files': sum(1 for f in self.files.values() if not f.should_include),
            'core_files': sum(1 for f in self.files.values() if f.category == FileCategory.CORE),
            'support_files': sum(1 for f in self.files.values() if f.category == FileCategory.SUPPORT),
            'config_files': sum(1 for f in self.files.values() if f.category == FileCategory.CONFIG),
            'data_files': sum(1 for f in self.files.values() if f.category == FileCategory.DATA),
            'test_files': sum(1 for f in self.files.values() if f.category == FileCategory.TEST),
        }
        return stats

    def get_include_patterns(self) -> list[str]:
        """Generate glob patterns for files to include."""
        patterns = set()

        for file_key, file_info in self.files.items():
            if file_info.should_include:
                # Add specific file
                patterns.add(file_key)

                # Add directory pattern for related files
                parent = str(Path(file_key).parent)
                if parent and parent != '.':
                    patterns.add(f"{parent}/*.py")

        return sorted(patterns)

    def get_exclude_patterns(self) -> list[str]:
        """Generate glob patterns for files to exclude."""
        patterns = set()

        for file_key, file_info in self.files.items():
            if not file_info.should_include:
                if file_info.category == FileCategory.DATA:
                    # Exclude data file extensions
                    suffix = Path(file_key).suffix
                    patterns.add(f"*{suffix}")
                elif file_info.category == FileCategory.TEST:
                    patterns.add("tests/")
                    patterns.add("*_test.py")
                    patterns.add("test_*.py")

        # Add standard excludes
        patterns.update([
            "__pycache__/",
            "*.pyc",
            ".git/",
            ".venv/",
            "venv/",
            "*.log",
            "*.tmp",
        ])

        return sorted(patterns)

    def generate_report(self) -> str:
        """Generate a human-readable analysis report."""
        result = self.analyze()

        lines = [
            "=" * 60,
            "SMART FILE ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Root Path: {self.root_path}",
            f"Minimum Relevance Threshold: {self.min_relevance}",
            "",
            "STATISTICS:",
            "-" * 40,
        ]

        for key, value in result.statistics.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")

        lines.extend([
            "",
            "CORE MODULES (High Relevance):",
            "-" * 40,
        ])

        for module in sorted(result.core_modules)[:20]:
            info = self.files.get(module)
            if info:
                lines.append(f"  [{info.relevance_score:.2f}] {module}")
                if info.keywords_found:
                    lines.append(f"         Keywords: {', '.join(info.keywords_found[:5])}")

        lines.extend([
            "",
            "RECOMMENDED EXCLUDES:",
            "-" * 40,
        ])

        for pattern in self.get_exclude_patterns()[:15]:
            lines.append(f"  - {pattern}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_codebase(
    root_path: str | Path,
    entry_points: list[str] | None = None,
) -> AnalysisResult:
    """Convenience function to analyze a codebase."""
    analyzer = SmartFileAnalyzer(
        root_path=Path(root_path),
        entry_points=entry_points,
    )
    return analyzer.analyze()


def get_trading_system_files(
    root_path: str | Path,
    min_relevance: float = 0.2,
) -> list[str]:
    """Get list of files related to the trading system."""
    analyzer = SmartFileAnalyzer(
        root_path=Path(root_path),
        min_relevance=min_relevance,
    )
    result = analyzer.analyze()
    return sorted(result.recommended_includes)


def generate_clone_config(root_path: str | Path) -> dict:
    """Generate recommended .clone.toml configuration."""
    analyzer = SmartFileAnalyzer(root_path=Path(root_path))
    result = analyzer.analyze()

    config = {
        'global': {
            'smart_analysis': True,
            'min_relevance_score': 0.2,
        },
        'global_excludes': analyzer.get_exclude_patterns(),
        'recommended_includes': list(result.recommended_includes)[:50],
        'statistics': result.statistics,
    }

    return config


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path.cwd()

    print(f"Analyzing: {root}\n")

    analyzer = SmartFileAnalyzer(
        root_path=root,
        entry_points=['cli', 'main', 'app'],
    )

    print(analyzer.generate_report())
