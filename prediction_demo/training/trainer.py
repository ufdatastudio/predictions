"""
Training orchestration for SpaCy NER models.

Handles model training, evaluation, and experiment management.
"""
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from loguru import logger
import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example

from prediction_demo.data.data_loader import (
    ENTITY_LABELS,
    prepare_training_data,
)
from prediction_demo.models.prediction_ner import (
    ModelType,
    PredictionNER,
    get_config_path,
)


def generate_model_name(model_type: ModelType, prefix: str = "prediction_ner") -> str:
    """
    Generate a default model name with timestamp.

    Args:
        model_type: Type of model.
        prefix: Prefix for the model name.

    Returns:
        Model name string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{model_type}_{timestamp}"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: ModelType = "cnn"
    model_name: str | None = None
    max_steps: int = 20000
    eval_frequency: int = 200
    dropout: float = 0.1
    batch_size: int = 1000
    learning_rate: float = 0.00005
    patience: int = 1600
    use_gpu: bool = False
    output_dir: Path = Path("models/prediction_ner")

    def get_model_name(self) -> str:
        """Get the model name, generating a default if not set."""
        if self.model_name:
            return self.model_name
        return generate_model_name(self.model_type)

    def to_overrides(self) -> list[str]:
        """Convert config to SpaCy CLI overrides."""
        overrides = [
            f"--training.max_steps={self.max_steps}",
            f"--training.eval_frequency={self.eval_frequency}",
            f"--training.dropout={self.dropout}",
            f"--training.patience={self.patience}",
            f"--training.optimizer.learn_rate.initial_rate={self.learning_rate}",
        ]
        if self.model_type == "cnn":
            overrides.append(f"--nlp.batch_size={self.batch_size}")
        return overrides


@dataclass
class TrainingResult:
    """Result from model training."""

    model_path: Path
    model_type: ModelType
    train_time: float
    best_score: float
    final_scores: dict
    config: TrainingConfig


@dataclass
class EvaluationResult:
    """Result from model evaluation."""

    overall_f1: float
    overall_precision: float
    overall_recall: float
    per_entity_scores: dict[str, dict[str, float]]
    total_entities: int
    correct_entities: int


class Trainer:
    """
    Orchestrates training and evaluation of SpaCy NER models.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            data_dir: Directory containing BIO-tagged train/dev/test files.
            output_dir: Directory for training outputs.
            progress_callback: Optional callback for progress updates.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.progress_callback = progress_callback

        self.processed_data_dir = self.output_dir / "processed_data"
        self.models_dir = self.output_dir / "models"

    def _report_progress(self, message: str, progress: float = 0.0) -> None:
        """Report progress via callback if available."""
        logger.info(message)
        if self.progress_callback:
            self.progress_callback(message, progress)

    def prepare_data(self) -> dict[str, Path]:
        """
        Prepare training data in SpaCy format.

        Returns:
            Dictionary mapping split names to DocBin file paths.
        """
        self._report_progress("Preparing training data...", 0.0)

        data_paths = prepare_training_data(
            data_dir=self.data_dir,
            output_dir=self.processed_data_dir,
        )

        self._report_progress("Data preparation complete", 0.1)
        return data_paths

    def train(self, config: TrainingConfig | None = None) -> TrainingResult:
        """
        Train a SpaCy NER model.

        Args:
            config: Training configuration. Uses defaults if None.

        Returns:
            TrainingResult with model path and metrics.
        """
        config = config or TrainingConfig()
        start_time = datetime.now()

        self._report_progress(f"Starting {config.model_type} model training", 0.1)

        data_paths = self.prepare_data()

        if "train" not in data_paths or "dev" not in data_paths:
            raise ValueError("Training and dev data required")

        model_name = config.get_model_name()
        model_output = self.models_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        config_path = get_config_path(config.model_type)

        cmd = [
            sys.executable,
            "-m",
            "spacy",
            "train",
            str(config_path),
            "--output",
            str(model_output),
            "--paths.train",
            str(data_paths["train"]),
            "--paths.dev",
            str(data_paths["dev"]),
        ]
        cmd.extend(config.to_overrides())

        if config.use_gpu:
            cmd.extend(["--gpu-id", "0"])

        self._report_progress("Running SpaCy training...", 0.2)
        logger.info(f"Training command: {' '.join(cmd)}")

        output_lines = []
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            best_score = 0.0
            for line in process.stdout:
                line_stripped = line.strip()
                output_lines.append(line_stripped)
                logger.info(line_stripped)
                if "LOSS" in line and "SCORE" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "SCORE":
                                score = float(parts[i + 1])
                                if score > best_score:
                                    best_score = score
                    except (ValueError, IndexError):
                        pass

            process.wait()

            if process.returncode != 0:
                error_output = "\n".join(output_lines[-50:])
                error_msg = (
                    f"Training failed with return code {process.returncode}\n\n"
                    f"Last 50 lines of output:\n{error_output}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise RuntimeError(f"Training error: {e}\n\nOutput:\n" + "\n".join(output_lines[-20:]))

        end_time = datetime.now()
        train_time = (end_time - start_time).total_seconds()

        best_model_path = model_output / "model-best"
        if not best_model_path.exists():
            best_model_path = model_output / "model-last"

        final_scores = {}
        if best_model_path.exists():
            eval_result = self.evaluate(best_model_path, data_paths.get("dev"))
            final_scores = {
                "f1": eval_result.overall_f1,
                "precision": eval_result.overall_precision,
                "recall": eval_result.overall_recall,
            }
            best_score = eval_result.overall_f1

        self._report_progress("Training complete", 1.0)

        return TrainingResult(
            model_path=best_model_path,
            model_type=config.model_type,
            train_time=train_time,
            best_score=best_score,
            final_scores=final_scores,
            config=config,
        )

    def evaluate(
        self, model_path: Path, test_data_path: Path | None = None
    ) -> EvaluationResult:
        """
        Evaluate a trained model.

        Args:
            model_path: Path to trained model.
            test_data_path: Path to test DocBin. Uses prepared test data if None.

        Returns:
            EvaluationResult with metrics.
        """
        self._report_progress("Evaluating model...", 0.0)

        nlp = spacy.load(model_path)

        if test_data_path is None:
            test_data_path = self.processed_data_dir / "test.spacy"
            if not test_data_path.exists():
                self.prepare_data()

        doc_bin = DocBin().from_disk(test_data_path)
        docs = list(doc_bin.get_docs(nlp.vocab))

        examples = []
        for gold_doc in docs:
            pred_doc = nlp(gold_doc.text)
            example = Example(pred_doc, gold_doc)
            examples.append(example)

        scorer = Scorer()
        scores = scorer.score(examples)

        ents_scores = scores.get("ents_per_type") or {}
        per_entity = {}
        for label in ENTITY_LABELS:
            label_scores = ents_scores.get(label) or {}
            per_entity[label] = {
                "precision": label_scores.get("p", 0.0),
                "recall": label_scores.get("r", 0.0),
                "f1": label_scores.get("f", 0.0),
            }

        total_entities = sum(len(doc.ents) for doc in docs)
        ents_f = scores.get("ents_f") or 0.0
        ents_p = scores.get("ents_p") or 0.0
        ents_r = scores.get("ents_r") or 0.0
        correct_entities = int(ents_f * total_entities)

        self._report_progress("Evaluation complete", 1.0)

        return EvaluationResult(
            overall_f1=ents_f,
            overall_precision=ents_p,
            overall_recall=ents_r,
            per_entity_scores=per_entity,
            total_entities=total_entities,
            correct_entities=correct_entities,
        )

    def compare_models(
        self, model_paths: dict[str, Path], test_data_path: Path | None = None
    ) -> dict[str, EvaluationResult]:
        """
        Compare multiple models on the same test set.

        Args:
            model_paths: Dictionary mapping model names to paths.
            test_data_path: Path to test data.

        Returns:
            Dictionary mapping model names to evaluation results.
        """
        results = {}
        for name, path in model_paths.items():
            logger.info(f"Evaluating {name}...")
            results[name] = self.evaluate(path, test_data_path)
        return results

    def grid_search(
        self,
        param_grid: dict[str, list],
        base_config: TrainingConfig | None = None,
        model_name_prefix: str = "grid_search",
    ) -> list[tuple[TrainingConfig, TrainingResult]]:
        """
        Perform grid search over hyperparameters.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
                        Supported parameters: dropout, learning_rate, max_steps,
                        batch_size, patience, model_type.
            base_config: Base configuration to modify. Uses defaults if None.
            model_name_prefix: Prefix for generated model names.

        Returns:
            List of (config, result) tuples sorted by best_score descending.
        """
        import itertools

        base_config = base_config or TrainingConfig()

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        total_runs = len(combinations)
        self._report_progress(f"Starting grid search with {total_runs} configurations", 0.0)

        results = []

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            config = TrainingConfig(
                model_type=params.get("model_type", base_config.model_type),
                model_name=f"{model_name_prefix}_{i+1:03d}",
                max_steps=params.get("max_steps", base_config.max_steps),
                eval_frequency=params.get("eval_frequency", base_config.eval_frequency),
                dropout=params.get("dropout", base_config.dropout),
                batch_size=params.get("batch_size", base_config.batch_size),
                learning_rate=params.get("learning_rate", base_config.learning_rate),
                patience=params.get("patience", base_config.patience),
                use_gpu=params.get("use_gpu", base_config.use_gpu),
            )

            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self._report_progress(
                f"Training {i+1}/{total_runs}: {param_str}",
                i / total_runs
            )

            try:
                result = self.train(config)
                results.append((config, result))
                logger.info(f"Config {i+1}: F1={result.best_score:.4f}")
            except Exception as e:
                logger.error(f"Config {i+1} failed: {e}")
                continue

        results.sort(key=lambda x: x[1].best_score, reverse=True)

        if results:
            best_config, best_result = results[0]
            self._report_progress(
                f"Grid search complete. Best F1: {best_result.best_score:.4f}",
                1.0
            )
        else:
            self._report_progress("Grid search complete. No successful runs.", 1.0)

        return results


@dataclass
class GridSearchResult:
    """Result from grid search."""

    results: list[tuple[TrainingConfig, TrainingResult]]
    best_config: TrainingConfig | None
    best_result: TrainingResult | None
    summary_df: "pd.DataFrame"

    @classmethod
    def from_results(
        cls, results: list[tuple[TrainingConfig, TrainingResult]]
    ) -> "GridSearchResult":
        """Create GridSearchResult from list of results."""
        import pandas as pd

        if not results:
            return cls(
                results=[],
                best_config=None,
                best_result=None,
                summary_df=pd.DataFrame(),
            )

        rows = []
        for config, result in results:
            rows.append({
                "model_name": config.get_model_name(),
                "model_type": config.model_type,
                "dropout": config.dropout,
                "learning_rate": config.learning_rate,
                "max_steps": config.max_steps,
                "batch_size": config.batch_size,
                "f1_score": result.best_score,
                "train_time": result.train_time,
                "model_path": str(result.model_path),
            })

        summary_df = pd.DataFrame(rows)
        best_config, best_result = results[0]

        return cls(
            results=results,
            best_config=best_config,
            best_result=best_result,
            summary_df=summary_df,
        )


def quick_train(
    data_dir: str | Path,
    output_dir: str | Path = "output",
    model_type: ModelType = "cnn",
    max_steps: int = 5000,
    use_gpu: bool = False,
) -> TrainingResult:
    """
    Quick training function for simple usage.

    Args:
        data_dir: Directory with BIO-tagged data.
        output_dir: Output directory.
        model_type: Model type to train.
        max_steps: Maximum training steps.
        use_gpu: Whether to use GPU.

    Returns:
        TrainingResult with trained model info.
    """
    config = TrainingConfig(
        model_type=model_type,
        max_steps=max_steps,
        use_gpu=use_gpu,
    )

    trainer = Trainer(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
    )

    return trainer.train(config)
