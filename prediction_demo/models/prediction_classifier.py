"""
Prediction classifier for determining full vs partial predictions.

Analyzes detected entities to classify texts as full predictions
(containing all 4 components) or partial predictions (missing components).
"""
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from prediction_demo.models.prediction_ner import NERResult, PredictionNER


class PredictionType(Enum):
    """Types of predictions based on component completeness."""

    FULL = "full"
    PARTIAL = "partial"
    NOT_A_PREDICTION = "not_a_prediction"


@dataclass
class PredictionComponents:
    """Container for prediction components."""

    source: str | None = None
    target: str | None = None
    date: str | None = None
    outcome: str | None = None

    @property
    def components_present(self) -> list[str]:
        """Get list of present component names."""
        present = []
        if self.source:
            present.append("source")
        if self.target:
            present.append("target")
        if self.date:
            present.append("date")
        if self.outcome:
            present.append("outcome")
        return present

    @property
    def components_missing(self) -> list[str]:
        """Get list of missing component names."""
        all_components = ["source", "target", "date", "outcome"]
        return [c for c in all_components if c not in self.components_present]

    @property
    def is_complete(self) -> bool:
        """Check if all components are present."""
        return len(self.components_present) == 4

    @property
    def component_count(self) -> int:
        """Get count of present components."""
        return len(self.components_present)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "date": self.date,
            "outcome": self.outcome,
            "components_present": self.components_present,
            "components_missing": self.components_missing,
            "is_complete": self.is_complete,
            "component_count": self.component_count,
        }


@dataclass
class ClassificationResult:
    """Result from prediction classification."""

    text: str
    prediction_type: PredictionType
    components: PredictionComponents
    confidence_score: float = 0.0
    ner_result: NERResult | None = None

    @property
    def is_prediction(self) -> bool:
        """Check if the text is classified as any type of prediction."""
        return self.prediction_type != PredictionType.NOT_A_PREDICTION

    @property
    def is_full_prediction(self) -> bool:
        """Check if the text is a full prediction."""
        return self.prediction_type == PredictionType.FULL

    @property
    def is_partial_prediction(self) -> bool:
        """Check if the text is a partial prediction."""
        return self.prediction_type == PredictionType.PARTIAL

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "prediction_type": self.prediction_type.value,
            "components": self.components.to_dict(),
            "confidence_score": self.confidence_score,
            "is_prediction": self.is_prediction,
            "is_full_prediction": self.is_full_prediction,
            "is_partial_prediction": self.is_partial_prediction,
        }


class PredictionClassifier:
    """
    Classifier for determining if text contains full or partial predictions.

    Uses NER results to analyze prediction components and classify text.
    """

    def __init__(self, ner_model: PredictionNER | None = None):
        """
        Initialize the classifier.

        Args:
            ner_model: NER model for entity extraction. If None, creates a new one.
        """
        self.ner_model = ner_model or PredictionNER()

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as full prediction, partial prediction, or not a prediction.

        Args:
            text: Input text to classify.

        Returns:
            ClassificationResult with prediction type and extracted components.
        """
        ner_result = self.ner_model.predict(text)
        components = self._extract_components(ner_result)
        prediction_type = self._determine_type(components)
        confidence = self._calculate_confidence(components, prediction_type)

        return ClassificationResult(
            text=text,
            prediction_type=prediction_type,
            components=components,
            confidence_score=confidence,
            ner_result=ner_result,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of ClassificationResult objects.
        """
        ner_results = self.ner_model.predict_batch(texts)
        results = []

        for text, ner_result in zip(texts, ner_results):
            components = self._extract_components(ner_result)
            prediction_type = self._determine_type(components)
            confidence = self._calculate_confidence(components, prediction_type)

            results.append(
                ClassificationResult(
                    text=text,
                    prediction_type=prediction_type,
                    components=components,
                    confidence_score=confidence,
                    ner_result=ner_result,
                )
            )

        return results

    def _extract_components(self, ner_result: NERResult) -> PredictionComponents:
        """
        Extract prediction components from NER result.

        Takes the first entity of each type if multiple are found.
        """
        return PredictionComponents(
            source=ner_result.sources[0].text if ner_result.sources else None,
            target=ner_result.targets[0].text if ner_result.targets else None,
            date=ner_result.dates[0].text if ner_result.dates else None,
            outcome=ner_result.outcomes[0].text if ner_result.outcomes else None,
        )

    def _determine_type(self, components: PredictionComponents) -> PredictionType:
        """Determine prediction type based on components."""
        if components.is_complete:
            return PredictionType.FULL
        elif components.component_count > 0:
            return PredictionType.PARTIAL
        else:
            return PredictionType.NOT_A_PREDICTION

    def _calculate_confidence(
        self, components: PredictionComponents, prediction_type: PredictionType
    ) -> float:
        """
        Calculate confidence score for the classification.

        Simple heuristic based on component count.
        """
        if prediction_type == PredictionType.NOT_A_PREDICTION:
            return 1.0

        return components.component_count / 4.0

    def get_summary(self, result: ClassificationResult) -> str:
        """
        Get a human-readable summary of the classification.

        Args:
            result: ClassificationResult to summarize.

        Returns:
            Summary string.
        """
        type_str = result.prediction_type.value.replace("_", " ").title()
        components = result.components

        summary = f"Classification: {type_str}\n"
        summary += f"Components found: {components.component_count}/4\n\n"

        if components.source:
            summary += f"Source (p_s): {components.source}\n"
        if components.target:
            summary += f"Target (p_t): {components.target}\n"
        if components.date:
            summary += f"Date (p_d): {components.date}\n"
        if components.outcome:
            summary += f"Outcome (p_o): {components.outcome}\n"

        if components.components_missing:
            summary += f"\nMissing: {', '.join(components.components_missing)}"

        return summary


def analyze_predictions(
    texts: list[str], ner_model: PredictionNER | None = None
) -> dict:
    """
    Analyze multiple texts and return statistics.

    Args:
        texts: List of texts to analyze.
        ner_model: NER model to use.

    Returns:
        Dictionary with statistics.
    """
    classifier = PredictionClassifier(ner_model)
    results = classifier.classify_batch(texts)

    stats = {
        "total": len(results),
        "full_predictions": sum(1 for r in results if r.is_full_prediction),
        "partial_predictions": sum(1 for r in results if r.is_partial_prediction),
        "not_predictions": sum(1 for r in results if not r.is_prediction),
        "component_coverage": {
            "source": sum(1 for r in results if r.components.source),
            "target": sum(1 for r in results if r.components.target),
            "date": sum(1 for r in results if r.components.date),
            "outcome": sum(1 for r in results if r.components.outcome),
        },
    }

    return stats
