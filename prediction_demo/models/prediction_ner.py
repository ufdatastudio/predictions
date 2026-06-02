"""
SpaCy NER model wrapper for prediction entity detection.

Provides a unified interface for loading, training, and using NER models
for detecting prediction components (source, target, date, outcome).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger
import spacy
from spacy.language import Language
from spacy.tokens import Doc


ENTITY_LABELS = ["P_SOURCE", "P_TARGET", "P_DATE", "P_OUTCOME"]

ModelType = Literal["cnn", "transformer"]


@dataclass
class Entity:
    """Represents a detected entity."""

    text: str
    label: str
    start: int
    end: int
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        """Convert entity to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class NERResult:
    """Result from NER inference."""

    text: str
    entities: list[Entity]

    @property
    def sources(self) -> list[Entity]:
        """Get source entities."""
        return [e for e in self.entities if e.label == "P_SOURCE"]

    @property
    def targets(self) -> list[Entity]:
        """Get target entities."""
        return [e for e in self.entities if e.label == "P_TARGET"]

    @property
    def dates(self) -> list[Entity]:
        """Get date entities."""
        return [e for e in self.entities if e.label == "P_DATE"]

    @property
    def outcomes(self) -> list[Entity]:
        """Get outcome entities."""
        return [e for e in self.entities if e.label == "P_OUTCOME"]

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "sources": [e.to_dict() for e in self.sources],
            "targets": [e.to_dict() for e in self.targets],
            "dates": [e.to_dict() for e in self.dates],
            "outcomes": [e.to_dict() for e in self.outcomes],
        }


class PredictionNER:
    """
    Wrapper for SpaCy NER models for prediction entity detection.

    Supports both CNN-based and transformer-based models.
    """

    def __init__(self, model_path: Path | str | None = None):
        """
        Initialize the NER model.

        Args:
            model_path: Path to a trained SpaCy model. If None, creates a blank model.
        """
        self.model_path = Path(model_path) if model_path else None
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        """Get the SpaCy language model, loading if necessary."""
        if self._nlp is None:
            self._nlp = self._load_model()
        return self._nlp

    def _load_model(self) -> Language:
        """Load the SpaCy model."""
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading model from {self.model_path}")
            return spacy.load(self.model_path)
        else:
            logger.info("Creating blank English model")
            return spacy.blank("en")

    def reload(self) -> None:
        """Reload the model from disk."""
        self._nlp = None
        _ = self.nlp

    def predict(self, text: str) -> NERResult:
        """
        Run NER prediction on text.

        Args:
            text: Input text to analyze.

        Returns:
            NERResult containing detected entities.
        """
        doc = self.nlp(text)
        entities = self._extract_entities(doc)
        return NERResult(text=text, entities=entities)

    def predict_batch(self, texts: list[str]) -> list[NERResult]:
        """
        Run NER prediction on multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of NERResult objects.
        """
        results = []
        for doc in self.nlp.pipe(texts):
            entities = self._extract_entities(doc)
            results.append(NERResult(text=doc.text, entities=entities))
        return results

    def _extract_entities(self, doc: Doc) -> list[Entity]:
        """Extract entities from a SpaCy Doc."""
        entities = []
        for ent in doc.ents:
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start,
                    end=ent.end,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )
        return entities

    def get_displacy_data(self, text: str) -> dict:
        """
        Get data for SpaCy displacy visualization.

        Args:
            text: Input text.

        Returns:
            Dictionary compatible with displacy.render().
        """
        doc = self.nlp(text)
        return {
            "text": doc.text,
            "ents": [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in doc.ents
            ],
            "title": None,
        }

    def render_entities(self, text: str, style: str = "ent") -> str:
        """
        Render entities as HTML using displacy.

        Args:
            text: Input text.
            style: Visualization style ("ent" or "dep").

        Returns:
            HTML string of the visualization.
        """
        from spacy import displacy

        doc = self.nlp(text)
        colors = {
            "P_SOURCE": "#7aecec",
            "P_TARGET": "#bfeeb7",
            "P_DATE": "#feca74",
            "P_OUTCOME": "#ff9561",
        }
        options = {"ents": ENTITY_LABELS, "colors": colors}
        return displacy.render(doc, style=style, options=options)


def create_blank_model_with_ner() -> Language:
    """
    Create a blank SpaCy model with NER component configured for prediction entities.

    Returns:
        SpaCy Language model with NER component.
    """
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in ENTITY_LABELS:
        ner.add_label(label)
    return nlp


def get_config_path(model_type: ModelType) -> Path:
    """
    Get the path to the config file for a model type.

    Args:
        model_type: Type of model ("cnn" or "transformer").

    Returns:
        Path to the config file.
    """
    config_dir = Path(__file__).parent.parent / "training" / "configs"
    if model_type == "cnn":
        return config_dir / "cnn_config.cfg"
    elif model_type == "transformer":
        return config_dir / "transformer_config.cfg"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
