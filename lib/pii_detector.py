#!/usr/bin/env python3
"""
pii_detector.py - Comprehensive PII Detection and Redaction Engine

A production-grade PII (Personally Identifiable Information) detection system
inspired by Microsoft Presidio but optimized for lightweight, fast scanning
without heavy NLP dependencies.

Features:
- Multi-category PII detection (SSN, phone, email, credit card, addresses, etc.)
- Context-aware confidence scoring
- Configurable sensitivity levels
- Financial data detection (account numbers, routing numbers)
- Trading-specific sensitive data (API keys, broker credentials)
- Checksum validation for credit cards (Luhn algorithm)
- Format normalization and pattern variations

References:
- Microsoft Presidio: https://github.com/microsoft/presidio
- NIST PII Guidelines: https://csrc.nist.gov/glossary/term/PII
- Yelp detect-secrets: https://github.com/Yelp/detect-secrets

Author: Clone System v2.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


class PIICategory(Enum):
    """Categories of PII for classification and filtering."""
    SSN = auto()
    PHONE = auto()
    EMAIL = auto()
    CREDIT_CARD = auto()
    BANK_ACCOUNT = auto()
    ROUTING_NUMBER = auto()
    ADDRESS = auto()
    DRIVERS_LICENSE = auto()
    PASSPORT = auto()
    DATE_OF_BIRTH = auto()
    IP_ADDRESS = auto()
    MAC_ADDRESS = auto()
    NAME = auto()

    # Financial/Trading specific
    API_KEY = auto()
    SECRET_KEY = auto()
    PASSWORD = auto()
    TOKEN = auto()
    PRIVATE_KEY = auto()
    BROKER_ACCOUNT = auto()

    # Generic
    NUMERIC_ID = auto()


class Sensitivity(Enum):
    """Detection sensitivity levels."""
    LOW = 1      # Only high-confidence matches
    MEDIUM = 2   # Balanced detection
    HIGH = 3     # Aggressive detection (more false positives)


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    category: PIICategory
    value: str
    start: int
    end: int
    confidence: float  # 0.0 to 1.0
    context: str = ""
    replacement: str = ""

    def __post_init__(self):
        if not self.replacement:
            self.replacement = f"<{self.category.name}>"


@dataclass
class PIIPattern:
    """A pattern for detecting a specific type of PII."""
    category: PIICategory
    pattern: re.Pattern
    base_confidence: float
    context_words: list[str] = field(default_factory=list)
    context_boost: float = 0.2
    validator: Callable[[str], bool] | None = None
    replacement: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.replacement:
            self.replacement = f"<{self.category.name}>"


# ─────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────────────────────

def luhn_checksum(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def validate_ssn(ssn: str) -> bool:
    """Validate SSN format and known invalid patterns."""
    digits = re.sub(r'\D', '', ssn)
    if len(digits) != 9:
        return False

    area, group, serial = digits[:3], digits[3:5], digits[5:]

    # Invalid area numbers
    if area == '000' or area == '666' or area[0] == '9':
        return False
    # Invalid group numbers
    if group == '00':
        return False
    # Invalid serial numbers
    if serial == '0000':
        return False
    # Known invalid SSNs (used in advertising, etc.)
    invalid_ssns = {'078051120', '219099999', '457555462'}
    if digits in invalid_ssns:
        return False

    return True


def validate_routing_number(routing: str) -> bool:
    """Validate ABA routing number checksum."""
    digits = re.sub(r'\D', '', routing)
    if len(digits) != 9:
        return False

    # ABA checksum: 3(d1 + d4 + d7) + 7(d2 + d5 + d8) + (d3 + d6 + d9) mod 10 = 0
    d = [int(x) for x in digits]
    checksum = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])
    return checksum % 10 == 0


def validate_ip_address(ip: str) -> bool:
    """Validate IPv4 address."""
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    try:
        for part in parts:
            num = int(part)
            if num < 0 or num > 255:
                return False
        # Exclude common non-sensitive IPs
        if ip.startswith('127.') or ip.startswith('0.') or ip == '255.255.255.255':
            return False
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PII Patterns Database
# ─────────────────────────────────────────────────────────────────────────────

PII_PATTERNS: list[PIIPattern] = [
    # ─────────────────────────────────────────────────────────────────────────
    # Social Security Numbers
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.SSN,
        pattern=re.compile(r'\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b'),
        base_confidence=0.7,
        context_words=['ssn', 'social', 'security', 'tax', 'taxpayer', 'tin', 'itin', 'ein'],
        context_boost=0.25,
        validator=validate_ssn,
        replacement="<SSN_REDACTED>",
        description="US Social Security Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Phone Numbers (US formats)
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.PHONE,
        pattern=re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?([2-9]\d{2})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b'
        ),
        base_confidence=0.6,
        context_words=['phone', 'tel', 'telephone', 'cell', 'mobile', 'fax', 'call', 'contact', 'number'],
        context_boost=0.3,
        replacement="<PHONE_REDACTED>",
        description="US Phone Number",
    ),

    # International phone format
    PIIPattern(
        category=PIICategory.PHONE,
        pattern=re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
        base_confidence=0.7,
        context_words=['phone', 'tel', 'international', 'mobile'],
        context_boost=0.2,
        replacement="<PHONE_INTL_REDACTED>",
        description="International Phone Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Email Addresses
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.EMAIL,
        pattern=re.compile(
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            re.IGNORECASE
        ),
        base_confidence=0.9,
        context_words=['email', 'mail', 'contact', 'address', 'send', 'reply'],
        context_boost=0.1,
        replacement="<EMAIL_REDACTED>",
        description="Email Address",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Credit Card Numbers
    # ─────────────────────────────────────────────────────────────────────────
    # Visa
    PIIPattern(
        category=PIICategory.CREDIT_CARD,
        pattern=re.compile(r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        base_confidence=0.85,
        context_words=['visa', 'card', 'credit', 'debit', 'payment', 'cc', 'cvv', 'expir'],
        context_boost=0.15,
        validator=luhn_checksum,
        replacement="<VISA_REDACTED>",
        description="Visa Card Number",
    ),
    # Mastercard
    PIIPattern(
        category=PIICategory.CREDIT_CARD,
        pattern=re.compile(r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        base_confidence=0.85,
        context_words=['mastercard', 'card', 'credit', 'debit', 'payment', 'cc'],
        context_boost=0.15,
        validator=luhn_checksum,
        replacement="<MASTERCARD_REDACTED>",
        description="Mastercard Number",
    ),
    # American Express
    PIIPattern(
        category=PIICategory.CREDIT_CARD,
        pattern=re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'),
        base_confidence=0.85,
        context_words=['amex', 'american express', 'card', 'credit'],
        context_boost=0.15,
        validator=luhn_checksum,
        replacement="<AMEX_REDACTED>",
        description="American Express Card Number",
    ),
    # Discover
    PIIPattern(
        category=PIICategory.CREDIT_CARD,
        pattern=re.compile(r'\b6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        base_confidence=0.85,
        context_words=['discover', 'card', 'credit', 'debit'],
        context_boost=0.15,
        validator=luhn_checksum,
        replacement="<DISCOVER_REDACTED>",
        description="Discover Card Number",
    ),
    # Generic 16-digit card
    PIIPattern(
        category=PIICategory.CREDIT_CARD,
        pattern=re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        base_confidence=0.5,
        context_words=['card', 'credit', 'debit', 'payment', 'cc', 'cvv', 'expir', 'pan'],
        context_boost=0.35,
        validator=luhn_checksum,
        replacement="<CARD_REDACTED>",
        description="Generic Credit Card Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Bank Account Numbers
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.BANK_ACCOUNT,
        pattern=re.compile(r'\b\d{8,17}\b'),
        base_confidence=0.3,
        context_words=['account', 'acct', 'bank', 'checking', 'savings', 'deposit', 'wire', 'ach'],
        context_boost=0.5,
        replacement="<BANK_ACCOUNT_REDACTED>",
        description="Bank Account Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Routing Numbers
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.ROUTING_NUMBER,
        pattern=re.compile(r'\b[0-3]\d{8}\b'),
        base_confidence=0.5,
        context_words=['routing', 'aba', 'transit', 'bank', 'wire', 'ach'],
        context_boost=0.4,
        validator=validate_routing_number,
        replacement="<ROUTING_REDACTED>",
        description="ABA Routing Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # IP Addresses
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.IP_ADDRESS,
        pattern=re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'),
        base_confidence=0.6,
        context_words=['ip', 'address', 'server', 'host', 'connect', 'network'],
        context_boost=0.3,
        validator=validate_ip_address,
        replacement="<IP_REDACTED>",
        description="IPv4 Address",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # MAC Addresses
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.MAC_ADDRESS,
        pattern=re.compile(
            r'\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
        ),
        base_confidence=0.8,
        context_words=['mac', 'hardware', 'device', 'network', 'ethernet'],
        context_boost=0.15,
        replacement="<MAC_REDACTED>",
        description="MAC Address",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Driver's License (US state formats - common patterns)
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.DRIVERS_LICENSE,
        pattern=re.compile(r'\b[A-Z]\d{7,8}\b'),
        base_confidence=0.4,
        context_words=['license', 'driver', 'dl', 'dmv', 'id', 'identification'],
        context_boost=0.45,
        replacement="<DL_REDACTED>",
        description="Driver's License Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Passport Numbers
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.PASSPORT,
        pattern=re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
        base_confidence=0.4,
        context_words=['passport', 'travel', 'visa', 'immigration', 'border'],
        context_boost=0.45,
        replacement="<PASSPORT_REDACTED>",
        description="Passport Number",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Date of Birth
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.DATE_OF_BIRTH,
        pattern=re.compile(
            r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b'
        ),
        base_confidence=0.5,
        context_words=['birth', 'dob', 'born', 'birthday', 'age', 'date of birth'],
        context_boost=0.4,
        replacement="<DOB_REDACTED>",
        description="Date of Birth (MM/DD/YYYY)",
    ),
    PIIPattern(
        category=PIICategory.DATE_OF_BIRTH,
        pattern=re.compile(
            r'\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b'
        ),
        base_confidence=0.5,
        context_words=['birth', 'dob', 'born', 'birthday', 'age'],
        context_boost=0.4,
        replacement="<DOB_REDACTED>",
        description="Date of Birth (YYYY-MM-DD)",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Street Addresses (US)
    # ─────────────────────────────────────────────────────────────────────────
    PIIPattern(
        category=PIICategory.ADDRESS,
        pattern=re.compile(
            r'\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl)\b',
            re.IGNORECASE
        ),
        base_confidence=0.7,
        context_words=['address', 'street', 'home', 'residence', 'mail', 'ship', 'deliver'],
        context_boost=0.2,
        replacement="<ADDRESS_REDACTED>",
        description="Street Address",
    ),

    # ZIP Codes (with context)
    PIIPattern(
        category=PIICategory.ADDRESS,
        pattern=re.compile(r'\b\d{5}(?:-\d{4})?\b'),
        base_confidence=0.3,
        context_words=['zip', 'postal', 'address', 'city', 'state', 'mail'],
        context_boost=0.5,
        replacement="<ZIP_REDACTED>",
        description="ZIP Code",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # API Keys and Secrets (Trading/Financial specific)
    # ─────────────────────────────────────────────────────────────────────────
    # AWS Keys
    PIIPattern(
        category=PIICategory.API_KEY,
        pattern=re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
        base_confidence=0.95,
        context_words=[],
        replacement="<AWS_ACCESS_KEY_REDACTED>",
        description="AWS Access Key ID",
    ),
    PIIPattern(
        category=PIICategory.API_KEY,
        pattern=re.compile(r'\bASIA[0-9A-Z]{16}\b'),
        base_confidence=0.95,
        context_words=[],
        replacement="<AWS_TEMP_KEY_REDACTED>",
        description="AWS Temporary Access Key",
    ),

    # GitHub Tokens
    PIIPattern(
        category=PIICategory.TOKEN,
        pattern=re.compile(r'\bghp_[A-Za-z0-9]{36,}\b'),
        base_confidence=0.95,
        context_words=[],
        replacement="<GITHUB_TOKEN_REDACTED>",
        description="GitHub Personal Access Token",
    ),
    PIIPattern(
        category=PIICategory.TOKEN,
        pattern=re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}\b'),
        base_confidence=0.95,
        context_words=[],
        replacement="<GITHUB_PAT_REDACTED>",
        description="GitHub Fine-grained PAT",
    ),

    # OpenAI Keys
    PIIPattern(
        category=PIICategory.API_KEY,
        pattern=re.compile(r'\bsk-[A-Za-z0-9]{20,}\b'),
        base_confidence=0.9,
        context_words=['openai', 'api', 'key'],
        context_boost=0.1,
        replacement="<OPENAI_KEY_REDACTED>",
        description="OpenAI API Key",
    ),

    # Generic API Keys (in config files)
    PIIPattern(
        category=PIICategory.API_KEY,
        pattern=re.compile(
            r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?'
        ),
        base_confidence=0.8,
        context_words=['api', 'key', 'secret', 'token'],
        context_boost=0.15,
        replacement=r'\1=<API_KEY_REDACTED>',
        description="Generic API Key",
    ),

    # Generic Secrets
    PIIPattern(
        category=PIICategory.SECRET_KEY,
        pattern=re.compile(
            r'(?i)(secret|secret[_-]?key)\s*[:=]\s*["\']?([A-Za-z0-9_\-/+=]{16,})["\']?'
        ),
        base_confidence=0.8,
        context_words=['secret', 'private', 'credential'],
        context_boost=0.15,
        replacement=r'\1=<SECRET_REDACTED>',
        description="Secret Key",
    ),

    # Passwords
    PIIPattern(
        category=PIICategory.PASSWORD,
        pattern=re.compile(
            r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']{6,})["\']?'
        ),
        base_confidence=0.85,
        context_words=['password', 'login', 'auth', 'credential'],
        context_boost=0.1,
        replacement='PASSWORD = auto()',
        description="Password",
    ),

    # Bearer Tokens (JWT)
    PIIPattern(
        category=PIICategory.TOKEN,
        pattern=re.compile(
            r'\b[Bb]earer\s+([A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+)\b'
        ),
        base_confidence=0.9,
        context_words=['authorization', 'auth', 'token', 'jwt'],
        context_boost=0.1,
        replacement="Bearer <JWT_REDACTED>",
        description="JWT Bearer Token",
    ),

    # Private Keys
    PIIPattern(
        category=PIICategory.PRIVATE_KEY,
        pattern=re.compile(
            r'-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----',
            re.MULTILINE
        ),
        base_confidence=0.99,
        context_words=[],
        replacement="<PRIVATE_KEY_BLOCK_REDACTED>",
        description="Private Key Block",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Broker/Trading Specific
    # ─────────────────────────────────────────────────────────────────────────
    # Interactive Brokers Account
    PIIPattern(
        category=PIICategory.BROKER_ACCOUNT,
        pattern=re.compile(r'\b[DUF]\d{7}\b'),
        base_confidence=0.6,
        context_words=['ibkr', 'interactive', 'broker', 'account', 'ib'],
        context_boost=0.35,
        replacement="<IBKR_ACCOUNT_REDACTED>",
        description="Interactive Brokers Account ID",
    ),

    # TD Ameritrade Account
    PIIPattern(
        category=PIICategory.BROKER_ACCOUNT,
        pattern=re.compile(r'\b\d{9}\b'),
        base_confidence=0.3,
        context_words=['td', 'ameritrade', 'tda', 'schwab', 'broker', 'account'],
        context_boost=0.5,
        replacement="<BROKER_ACCOUNT_REDACTED>",
        description="Broker Account Number",
    ),

    # Databento API Key
    PIIPattern(
        category=PIICategory.API_KEY,
        pattern=re.compile(r'\bdb-[A-Za-z0-9]{32}\b'),
        base_confidence=0.95,
        context_words=['databento', 'api'],
        replacement="<DATABENTO_KEY_REDACTED>",
        description="Databento API Key",
    ),

    # Polygon API Key
    PIIPattern(
        category=PIICategory.API_KEY,
        pattern=re.compile(r'\b[A-Za-z0-9_]{32}\b'),
        base_confidence=0.3,
        context_words=['polygon', 'api', 'key'],
        context_boost=0.55,
        replacement="<POLYGON_KEY_REDACTED>",
        description="Polygon.io API Key",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# PII Detector Class
# ─────────────────────────────────────────────────────────────────────────────

class PIIDetector:
    """
    Comprehensive PII detection engine with context-aware confidence scoring.

    Usage:
        detector = PIIDetector(sensitivity=Sensitivity.MEDIUM)
        matches = detector.scan(text)
        redacted_text = detector.redact(text)
    """

    def __init__(
        self,
        sensitivity: Sensitivity = Sensitivity.MEDIUM,
        categories: set[PIICategory] | None = None,
        custom_patterns: list[PIIPattern] | None = None,
        context_window: int = 50,
    ):
        """
        Initialize the PII detector.

        Args:
            sensitivity: Detection sensitivity level
            categories: Optional set of categories to detect (None = all)
            custom_patterns: Additional custom patterns to include
            context_window: Characters to examine for context words
        """
        self.sensitivity = sensitivity
        self.categories = categories
        self.context_window = context_window

        # Build pattern list
        self.patterns = list(PII_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        # Filter by categories if specified
        if categories:
            self.patterns = [p for p in self.patterns if p.category in categories]

        # Set confidence threshold based on sensitivity
        self.confidence_threshold = {
            Sensitivity.LOW: 0.8,
            Sensitivity.MEDIUM: 0.6,
            Sensitivity.HIGH: 0.4,
        }[sensitivity]

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Extract context around a match."""
        ctx_start = max(0, start - self.context_window)
        ctx_end = min(len(text), end + self.context_window)
        return text[ctx_start:ctx_end].lower()

    def _calculate_confidence(
        self,
        pattern: PIIPattern,
        match: re.Match,
        text: str,
    ) -> float:
        """Calculate confidence score with context boosting."""
        confidence = pattern.base_confidence

        # Get context
        context = self._get_context(text, match.start(), match.end())

        # Boost confidence if context words are present
        if pattern.context_words:
            for word in pattern.context_words:
                if word.lower() in context:
                    confidence += pattern.context_boost
                    break  # Only boost once

        # Apply validator if present
        if pattern.validator:
            matched_text = match.group(0) if match.lastindex is None else match.group(1)
            if not pattern.validator(matched_text):
                confidence *= 0.3  # Significant penalty for failed validation

        return min(confidence, 1.0)

    def scan(self, text: str) -> list[PIIMatch]:
        """
        Scan text for PII matches.

        Args:
            text: Text to scan

        Returns:
            List of PIIMatch objects above confidence threshold
        """
        matches: list[PIIMatch] = []
        seen_ranges: list[tuple[int, int]] = []

        def overlaps_existing(start: int, end: int) -> bool:
            """Check if a range overlaps with any existing match."""
            for s, e in seen_ranges:
                if start < e and end > s:  # Ranges overlap
                    return True
            return False

        for pattern in self.patterns:
            for match in pattern.pattern.finditer(text):
                start, end = match.start(), match.end()

                # Skip if this match overlaps with an existing one
                if overlaps_existing(start, end):
                    continue

                confidence = self._calculate_confidence(pattern, match, text)

                if confidence >= self.confidence_threshold:
                    seen_ranges.append((start, end))
                    matches.append(PIIMatch(
                        category=pattern.category,
                        value=match.group(0),
                        start=start,
                        end=end,
                        confidence=confidence,
                        context=self._get_context(text, start, end),
                        replacement=pattern.replacement,
                    ))

        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches

    def redact(self, text: str, matches: list[PIIMatch] | None = None) -> str:
        """
        Redact PII from text.

        Args:
            text: Text to redact
            matches: Optional pre-computed matches (will scan if not provided)

        Returns:
            Text with PII replaced by placeholders
        """
        if matches is None:
            matches = self.scan(text)

        if not matches:
            return text

        # Process in reverse order to preserve positions
        result = text
        for match in reversed(matches):
            result = result[:match.start] + match.replacement + result[match.end:]

        return result

    def scan_file(self, filepath: str) -> list[PIIMatch]:
        """Scan a file for PII."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.scan(content)
        except Exception:
            return []

    def redact_file(self, filepath: str, output_path: str | None = None) -> tuple[str, list[PIIMatch]]:
        """
        Redact PII from a file.

        Args:
            filepath: Path to input file
            output_path: Optional output path (modifies in place if None)

        Returns:
            Tuple of (redacted content, list of matches)
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        matches = self.scan(content)
        redacted = self.redact(content, matches)

        if output_path or matches:
            out = output_path or filepath
            with open(out, 'w', encoding='utf-8') as f:
                f.write(redacted)

        return redacted, matches

    def get_summary(self, matches: list[PIIMatch]) -> dict[str, int]:
        """Get a summary count of matches by category."""
        summary: dict[str, int] = {}
        for match in matches:
            key = match.category.name
            summary[key] = summary.get(key, 0) + 1
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def scan_text(text: str, sensitivity: Sensitivity = Sensitivity.MEDIUM) -> list[PIIMatch]:
    """Convenience function to scan text for PII."""
    return PIIDetector(sensitivity=sensitivity).scan(text)


def redact_text(text: str, sensitivity: Sensitivity = Sensitivity.MEDIUM) -> str:
    """Convenience function to redact PII from text."""
    return PIIDetector(sensitivity=sensitivity).redact(text)


def scan_file(filepath: str, sensitivity: Sensitivity = Sensitivity.MEDIUM) -> list[PIIMatch]:
    """Convenience function to scan a file for PII."""
    return PIIDetector(sensitivity=sensitivity).scan_file(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# CLI for testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Test data with synthetic PII examples for demonstration
    # Note: All data below is fake/synthetic for testing purposes only
    test_text = """
    Contact John Smith at <EMAIL_REDACTED> or call <PHONE_REDACTED>.
    His SSN is 078-05-1120 and he lives at <ADDRESS_REDACTED>, Anytown.

    Payment info:
    - Card: <VISA_REDACTED>
    - Account: <BANK_ACCOUNT_REDACTED>
    - Routing: <BANK_ACCOUNT_REDACTED>

    API Keys:
    - AWS: <AWS_ACCESS_KEY_REDACTED>
    - GitHub: <GITHUB_TOKEN_REDACTED>
    - OpenAI: <OPENAI_KEY_REDACTED>

    Broker account: <IBKR_ACCOUNT_REDACTED>
    IP Address: <IP_REDACTED>
    """

    detector = PIIDetector(sensitivity=Sensitivity.MEDIUM)
    matches = detector.scan(test_text)

    print("=== PII Detection Results ===\n")
    for match in matches:
        print(f"[{match.category.name}] '{match.value}' (confidence: {match.confidence:.2f})")

    print("\n=== Summary ===")
    print(detector.get_summary(matches))

    print("\n=== Redacted Text ===")
    print(detector.redact(test_text, matches))
