"""
Data loader for BIO-tagged prediction data.

Parses BIO-tagged training data and converts to SpaCy DocBin format for training.
"""
from pathlib import Path
from typing import Generator

from loguru import logger
import spacy
from spacy.tokens import Doc, DocBin


LABEL_MAP = {
    "B-p_s": "P_SOURCE",
    "I-p_s": "P_SOURCE",
    "B-p_t": "P_TARGET",
    "I-p_t": "P_TARGET",
    "B-p_d": "P_DATE",
    "I-p_d": "P_DATE",
    "B-p_o": "P_OUTCOME",
    "I-p_o": "P_OUTCOME",
    "O": "O",
}

ENTITY_LABELS = ["P_SOURCE", "P_TARGET", "P_DATE", "P_OUTCOME"]


def parse_bio_file(file_path: Path) -> Generator[list[tuple[str, str]], None, None]:
    """
    Parse a BIO-tagged file and yield sentences as lists of (token, label) tuples.

    Args:
        file_path: Path to the BIO-tagged file.

    Yields:
        List of (token, label) tuples for each sentence.
    """
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            parts = line.split("\t")

            if len(parts) >= 3:
                token = parts[1].strip()
                label = parts[2].strip()

                if not token or not label:
                    if current_sentence:
                        yield current_sentence
                        current_sentence = []
                    continue

                current_sentence.append((token, label))
            else:
                if current_sentence:
                    yield current_sentence
                    current_sentence = []

    if current_sentence:
        yield current_sentence


def bio_to_spans(
    tokens: list[str], labels: list[str]
) -> list[tuple[int, int, str]]:
    """
    Convert BIO labels to character spans.

    Args:
        tokens: List of tokens.
        labels: List of BIO labels.

    Returns:
        List of (start_char, end_char, label) tuples.
    """
    spans = []
    current_entity = None
    current_start = 0
    char_offset = 0

    for i, (token, label) in enumerate(zip(tokens, labels)):
        token_start = char_offset
        token_end = char_offset + len(token)

        if label.startswith("B-"):
            if current_entity is not None:
                spans.append(
                    (current_start, char_offset - 1, LABEL_MAP[current_entity])
                )
            current_entity = label
            current_start = token_start

        elif label.startswith("I-"):
            if current_entity is None:
                current_entity = label.replace("I-", "B-")
                current_start = token_start
            elif label.replace("I-", "") != current_entity.replace("B-", "").replace(
                "I-", ""
            ):
                spans.append(
                    (current_start, char_offset - 1, LABEL_MAP[current_entity])
                )
                current_entity = label.replace("I-", "B-")
                current_start = token_start

        else:
            if current_entity is not None:
                spans.append(
                    (current_start, char_offset - 1, LABEL_MAP[current_entity])
                )
                current_entity = None

        char_offset = token_end + 1

    if current_entity is not None:
        spans.append((current_start, char_offset - 1, LABEL_MAP[current_entity]))

    return spans


def create_training_example(
    nlp: spacy.Language, tokens: list[str], labels: list[str]
) -> Doc | None:
    """
    Create a SpaCy Doc with entity annotations from tokens and labels.

    Args:
        nlp: SpaCy language model.
        tokens: List of tokens.
        labels: List of BIO labels.

    Returns:
        SpaCy Doc with entity annotations, or None if invalid.
    """
    text = " ".join(tokens)
    doc = nlp.make_doc(text)

    spans = bio_to_spans(tokens, labels)

    ents = []
    for start, end, label in spans:
        if label == "O":
            continue
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is not None:
            ents.append(span)

    try:
        doc.ents = ents
        return doc
    except ValueError as e:
        logger.warning(f"Could not set entities for text: {text[:50]}... Error: {e}")
        return None


def load_data_as_docbin(
    file_path: Path, nlp: spacy.Language | None = None
) -> DocBin:
    """
    Load BIO-tagged data and convert to SpaCy DocBin format.

    Args:
        file_path: Path to the BIO-tagged file.
        nlp: SpaCy language model. If None, creates a blank English model.

    Returns:
        SpaCy DocBin containing annotated documents.
    """
    if nlp is None:
        nlp = spacy.blank("en")

    doc_bin = DocBin()
    sentence_count = 0
    error_count = 0

    for sentence in parse_bio_file(file_path):
        tokens = [t[0] for t in sentence]
        labels = [t[1] for t in sentence]

        doc = create_training_example(nlp, tokens, labels)
        if doc is not None:
            doc_bin.add(doc)
            sentence_count += 1
        else:
            error_count += 1

    logger.info(
        f"Loaded {sentence_count} sentences from {file_path.name} "
        f"({error_count} errors)"
    )
    return doc_bin


def save_docbin(doc_bin: DocBin, output_path: Path) -> None:
    """
    Save DocBin to disk.

    Args:
        doc_bin: SpaCy DocBin to save.
        output_path: Path to save the DocBin.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(output_path)
    logger.info(f"Saved DocBin to {output_path}")


def prepare_training_data(
    data_dir: Path,
    output_dir: Path,
    nlp: spacy.Language | None = None,
) -> dict[str, Path]:
    """
    Prepare training, dev, and test data in SpaCy DocBin format.

    Args:
        data_dir: Directory containing train, dev, test files.
        output_dir: Directory to save DocBin files.
        nlp: SpaCy language model.

    Returns:
        Dictionary mapping split names to output paths.
    """
    if nlp is None:
        nlp = spacy.blank("en")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    for split in ["train", "dev", "test"]:
        input_path = data_dir / split
        if not input_path.exists():
            logger.warning(f"Split {split} not found at {input_path}")
            continue

        doc_bin = load_data_as_docbin(input_path, nlp)
        output_path = output_dir / f"{split}.spacy"
        save_docbin(doc_bin, output_path)
        output_paths[split] = output_path

    return output_paths


def get_data_stats(file_path: Path) -> dict:
    """
    Get statistics about the BIO-tagged data.

    Args:
        file_path: Path to the BIO-tagged file.

    Returns:
        Dictionary with data statistics.
    """
    stats = {
        "total_sentences": 0,
        "total_tokens": 0,
        "entity_counts": {label: 0 for label in ENTITY_LABELS},
    }

    for sentence in parse_bio_file(file_path):
        stats["total_sentences"] += 1
        stats["total_tokens"] += len(sentence)

        current_entity = None
        for _, label in sentence:
            if label.startswith("B-"):
                mapped_label = LABEL_MAP[label]
                if mapped_label in stats["entity_counts"]:
                    stats["entity_counts"][mapped_label] += 1
                current_entity = label
            elif label.startswith("I-"):
                if current_entity is None:
                    mapped_label = LABEL_MAP[label]
                    if mapped_label in stats["entity_counts"]:
                        stats["entity_counts"][mapped_label] += 1
            else:
                current_entity = None

    return stats
