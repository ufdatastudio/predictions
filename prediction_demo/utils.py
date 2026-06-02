"""
Utility functions for the Prediction NER Demo.
"""
from pathlib import Path
from typing import Iterator

from loguru import logger


def find_project_root() -> Path:
    """
    Find the project root directory.

    Returns:
        Path to project root.
    """
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def iter_sentences(text: str) -> Iterator[str]:
    """
    Iterate over sentences in text.

    Simple sentence splitter based on punctuation.

    Args:
        text: Input text.

    Yields:
        Individual sentences.
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            yield sentence


def format_entity_spans(text: str, entities: list[dict]) -> str:
    """
    Format text with entity annotations.

    Args:
        text: Original text.
        entities: List of entity dictionaries with start_char, end_char, label.

    Returns:
        Formatted string with entity markers.
    """
    sorted_entities = sorted(entities, key=lambda x: x["start_char"], reverse=True)

    result = text
    for entity in sorted_entities:
        start = entity["start_char"]
        end = entity["end_char"]
        label = entity["label"]
        entity_text = result[start:end]
        result = f"{result[:start]}[{entity_text}]({label}){result[end:]}"

    return result


def calculate_iou(span1: tuple[int, int], span2: tuple[int, int]) -> float:
    """
    Calculate Intersection over Union for two spans.

    Args:
        span1: (start, end) tuple.
        span2: (start, end) tuple.

    Returns:
        IoU score between 0 and 1.
    """
    start1, end1 = span1
    start2, end2 = span2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    union = (end1 - start1) + (end2 - start2) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Setup loguru logging configuration.

    Args:
        log_level: Logging level.
        log_file: Optional log file path.
    """
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    if log_file:
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
        )
