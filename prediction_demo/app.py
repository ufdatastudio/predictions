"""
Streamlit interface for Prediction NER Demo.

Provides tabs for inference, training, evaluation, and batch processing.
"""
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st
from loguru import logger
from spacy import displacy

from prediction_demo.data.data_loader import get_data_stats, ENTITY_LABELS
from prediction_demo.models.prediction_ner import PredictionNER
from prediction_demo.models.prediction_classifier import (
    PredictionClassifier,
    PredictionType,
)
from prediction_demo.training.trainer import Trainer, TrainingConfig


def check_spacy_transformers_available() -> bool:
    """Check if spacy-transformers is installed."""
    try:
        import spacy_transformers
        return True
    except ImportError:
        return False


SPACY_TRANSFORMERS_AVAILABLE = check_spacy_transformers_available()


def init_session_state():
    """Initialize session state variables."""
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "log_handler_added" not in st.session_state:
        st.session_state.log_handler_added = False


def add_log(message: str, level: str = "INFO"):
    """Add a log message to session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {level}: {message}")
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]


def setup_logging():
    """Setup loguru to capture logs for display."""
    if not st.session_state.log_handler_added:
        def streamlit_sink(message):
            record = message.record
            add_log(record["message"], record["level"].name)

        logger.add(streamlit_sink, format="{message}", level="DEBUG")
        st.session_state.log_handler_added = True


def clear_logs():
    """Clear all logs."""
    st.session_state.logs = []


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tagging" / "official"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"


ENTITY_COLORS = {
    "P_SOURCE": "#7aecec",
    "P_TARGET": "#bfeeb7",
    "P_DATE": "#feca74",
    "P_OUTCOME": "#ff9561",
}


def get_available_models() -> dict[str, Path]:
    """Get dictionary of available trained models."""
    models = {}
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                best_path = model_dir / "model-best"
                last_path = model_dir / "model-last"
                if best_path.exists():
                    models[model_dir.name] = best_path
                elif last_path.exists():
                    models[model_dir.name] = last_path
    return models


@st.cache_resource
def load_ner_model(model_path: str | None) -> PredictionNER:
    """Load NER model with caching."""
    if model_path and Path(model_path).exists():
        return PredictionNER(model_path)
    return PredictionNER()


def render_entities_html(text: str, entities: list[dict]) -> str:
    """Render entities as HTML using SpaCy displacy."""
    doc_data = {
        "text": text,
        "ents": [
            {"start": e["start_char"], "end": e["end_char"], "label": e["label"]}
            for e in entities
        ],
        "title": None,
    }
    options = {"ents": ENTITY_LABELS, "colors": ENTITY_COLORS}
    return displacy.render(doc_data, style="ent", manual=True, options=options)


def inference_tab():
    """Inference tab for single text analysis."""
    st.header("Prediction Entity Detection")

    available_models = get_available_models()

    col1, col2 = st.columns([3, 1])
    with col1:
        if available_models:
            model_choice = st.selectbox(
                "Select Model",
                options=["No model (blank)"] + list(available_models.keys()),
                index=0,
            )
        else:
            model_choice = "No model (blank)"
            st.info("No trained models found. Train a model in the Training tab.")

    model_path = None
    if model_choice != "No model (blank)" and model_choice in available_models:
        model_path = str(available_models[model_choice])

    ner_model = load_ner_model(model_path)
    classifier = PredictionClassifier(ner_model)

    sample_texts = [
        "Morgan Stanley predicts that on September 15, 2025, the S&P 500 composite index will likely rise.",
        "Goldman Sachs speculates that the operating cash flow at Microsoft will likely increase.",
        "The Federal Reserve expects inflation to decrease by next year.",
        "According to Apple, the projected revenue at Amazon will likely fall in Q4 2026.",
    ]

    selected_sample = st.selectbox(
        "Try a sample",
        options=["Custom input"] + sample_texts,
        index=0,
    )

    if selected_sample == "Custom input":
        text_input = st.text_area(
            "Enter text to analyze",
            height=100,
            placeholder="Enter a prediction statement...",
        )
    else:
        text_input = st.text_area(
            "Enter text to analyze",
            value=selected_sample,
            height=100,
        )

    if st.button("Analyze", type="primary") and text_input:
        with st.spinner("Analyzing..."):
            result = classifier.classify(text_input)

        st.subheader("Entity Visualization")
        if result.ner_result and result.ner_result.entities:
            entities_data = [e.to_dict() for e in result.ner_result.entities]
            html = render_entities_html(text_input, entities_data)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No entities detected")

        st.subheader("Classification Result")
        col1, col2, col3 = st.columns(3)

        with col1:
            pred_type = result.prediction_type.value.replace("_", " ").title()
            if result.prediction_type == PredictionType.FULL:
                st.success(f"Type: {pred_type}")
            elif result.prediction_type == PredictionType.PARTIAL:
                st.warning(f"Type: {pred_type}")
            else:
                st.error(f"Type: {pred_type}")

        with col2:
            st.metric(
                "Components Found",
                f"{result.components.component_count}/4",
            )

        with col3:
            st.metric(
                "Confidence",
                f"{result.confidence_score:.0%}",
            )

        st.subheader("Extracted Components")
        components = result.components

        comp_data = {
            "Component": ["Source (p_s)", "Target (p_t)", "Date (p_d)", "Outcome (p_o)"],
            "Value": [
                components.source or "-",
                components.target or "-",
                components.date or "-",
                components.outcome or "-",
            ],
            "Status": [
                "Found" if components.source else "Missing",
                "Found" if components.target else "Missing",
                "Found" if components.date else "Missing",
                "Found" if components.outcome else "Missing",
            ],
        }
        st.table(pd.DataFrame(comp_data))


def training_tab():
    """Training tab for model training."""
    st.header("Model Training")

    col1, col2 = st.columns(2)

    with col1:
        if SPACY_TRANSFORMERS_AVAILABLE:
            model_type = st.selectbox(
                "Model Type",
                options=["cnn", "transformer"],
                index=0,
                help="CNN is faster, Transformer has higher accuracy",
            )
        else:
            model_type = st.selectbox(
                "Model Type",
                options=["cnn"],
                index=0,
                help="CNN model (transformer requires spacy-transformers)",
            )
            st.caption(
                "Transformer model unavailable. Install with: "
                "`uv add spacy-transformers` (requires transformers<4.50)"
            )

    with col2:
        max_steps = st.number_input(
            "Max Training Steps",
            min_value=100,
            max_value=50000,
            value=5000,
            step=500,
        )

    model_name = st.text_input(
        "Model Name (optional)",
        value="",
        placeholder="Leave empty for auto-generated name",
        help="Custom name for the model. If empty, generates: prediction_ner_{type}_{timestamp}",
    )

    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=0.001,
                value=0.00005,
                format="%.5f",
            )
        with col2:
            eval_frequency = st.number_input(
                "Eval Frequency",
                min_value=50,
                max_value=1000,
                value=200,
            )
            patience = st.number_input(
                "Patience",
                min_value=100,
                max_value=5000,
                value=1600,
            )

    use_gpu = st.checkbox("Use GPU (if available)", value=False)

    st.divider()
    enable_grid_search = st.checkbox(
        "Enable Grid Search",
        value=False,
        help="Train multiple models with different hyperparameters",
    )

    if enable_grid_search:
        st.subheader("Grid Search Configuration")
        st.caption("Select multiple values for each parameter to search over.")

        grid_col1, grid_col2 = st.columns(2)
        with grid_col1:
            grid_dropout = st.multiselect(
                "Dropout Values",
                options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
                default=[0.1],
            )
            grid_learning_rate = st.multiselect(
                "Learning Rate Values",
                options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                default=[0.00005],
                format_func=lambda x: f"{x:.5f}",
            )
        with grid_col2:
            grid_max_steps = st.multiselect(
                "Max Steps Values",
                options=[1000, 2000, 5000, 10000, 20000],
                default=[5000],
            )
            grid_batch_size = st.multiselect(
                "Batch Size Values",
                options=[100, 500, 1000, 2000],
                default=[1000],
            )

        total_combinations = (
            len(grid_dropout) * len(grid_learning_rate) *
            len(grid_max_steps) * len(grid_batch_size)
        )
        st.info(f"Total configurations to train: {total_combinations}")

    if DATA_DIR.exists():
        st.subheader("Data Statistics")
        stats = get_data_stats(DATA_DIR / "train")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentences", stats["total_sentences"])
        with col2:
            st.metric("Tokens", stats["total_tokens"])
        with col3:
            total_ents = sum(stats["entity_counts"].values())
            st.metric("Entities", total_ents)
    else:
        st.warning(f"Training data not found at {DATA_DIR}")

    if enable_grid_search:
        if st.button("Start Grid Search", type="primary"):
            param_grid = {
                "dropout": grid_dropout,
                "learning_rate": grid_learning_rate,
                "max_steps": grid_max_steps,
                "batch_size": grid_batch_size,
            }

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(message: str, progress: float):
                progress_bar.progress(min(progress, 1.0))
                status_text.text(message)

            trainer = Trainer(
                data_dir=DATA_DIR,
                output_dir=OUTPUT_DIR,
                progress_callback=progress_callback,
            )

            base_config = TrainingConfig(
                model_type=model_type,
                eval_frequency=eval_frequency,
                patience=patience,
                use_gpu=use_gpu,
            )

            prefix = model_name if model_name else "grid_search"

            with st.spinner("Grid search in progress..."):
                try:
                    add_log(f"Starting grid search with {total_combinations} configurations", "INFO")
                    results = trainer.grid_search(
                        param_grid=param_grid,
                        base_config=base_config,
                        model_name_prefix=prefix,
                    )

                    if results:
                        from prediction_demo.training.trainer import GridSearchResult
                        grid_result = GridSearchResult.from_results(results)

                        st.success(f"Grid search complete! {len(results)} models trained.")
                        add_log(f"Grid search complete! Best F1: {grid_result.best_result.best_score:.2%}", "INFO")

                        st.subheader("Results Summary")
                        st.dataframe(grid_result.summary_df)

                        if grid_result.best_config:
                            st.subheader("Best Configuration")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Best F1", f"{grid_result.best_result.best_score:.2%}")
                            with col2:
                                st.metric("Dropout", grid_result.best_config.dropout)
                            with col3:
                                st.metric("Learning Rate", f"{grid_result.best_config.learning_rate:.5f}")
                            with col4:
                                st.metric("Max Steps", grid_result.best_config.max_steps)

                        st.cache_resource.clear()
                    else:
                        st.error("No successful training runs in grid search.")

                except Exception as e:
                    error_msg = str(e)
                    add_log(f"Grid search failed: {error_msg}", "ERROR")
                    st.error("Grid search failed!")
                    with st.expander("Error Details", expanded=True):
                        st.code(error_msg, language="text")
    else:
        if st.button("Start Training", type="primary"):
            config = TrainingConfig(
                model_type=model_type,
                model_name=model_name if model_name else None,
                max_steps=max_steps,
                dropout=dropout,
                learning_rate=learning_rate,
                eval_frequency=eval_frequency,
                patience=patience,
                use_gpu=use_gpu,
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(message: str, progress: float):
                progress_bar.progress(progress)
                status_text.text(message)

            trainer = Trainer(
                data_dir=DATA_DIR,
                output_dir=OUTPUT_DIR,
                progress_callback=progress_callback,
            )

            with st.spinner("Training in progress..."):
                try:
                    add_log(f"Starting {model_type} training with {max_steps} steps", "INFO")
                    result = trainer.train(config)

                    st.success("Training complete!")
                    add_log(f"Training complete! Best F1: {result.best_score:.2%}", "INFO")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best F1 Score", f"{result.best_score:.2%}")
                    with col2:
                        st.metric("Training Time", f"{result.train_time:.1f}s")
                    with col3:
                        st.metric("Model Path", str(result.model_path.name))

                    st.cache_resource.clear()

                except Exception as e:
                    error_msg = str(e)
                    add_log(f"Training failed: {error_msg}", "ERROR")
                    st.error("Training failed!")
                    with st.expander("Error Details", expanded=True):
                        st.code(error_msg, language="text")


def evaluation_tab():
    """Evaluation tab for model comparison."""
    st.header("Model Evaluation")

    available_models = get_available_models()

    if not available_models:
        st.warning("No trained models found. Train a model first.")
        return

    selected_models = st.multiselect(
        "Select models to evaluate",
        options=list(available_models.keys()),
        default=list(available_models.keys())[:2] if len(available_models) >= 2 else list(available_models.keys()),
    )

    if st.button("Evaluate", type="primary") and selected_models:
        trainer = Trainer(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)

        results = {}
        progress = st.progress(0)

        for i, model_name in enumerate(selected_models):
            st.text(f"Evaluating {model_name}...")
            model_path = available_models[model_name]
            results[model_name] = trainer.evaluate(model_path)
            progress.progress((i + 1) / len(selected_models))

        st.subheader("Overall Metrics")
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                "Model": name,
                "F1": f"{result.overall_f1:.2%}",
                "Precision": f"{result.overall_precision:.2%}",
                "Recall": f"{result.overall_recall:.2%}",
            })
        st.table(pd.DataFrame(metrics_data))

        st.subheader("Per-Entity Metrics")
        for model_name, result in results.items():
            with st.expander(f"{model_name} - Entity Breakdown"):
                entity_data = []
                for label, scores in result.per_entity_scores.items():
                    entity_data.append({
                        "Entity": label,
                        "F1": f"{scores['f1']:.2%}",
                        "Precision": f"{scores['precision']:.2%}",
                        "Recall": f"{scores['recall']:.2%}",
                    })
                st.table(pd.DataFrame(entity_data))


def batch_processing_tab():
    """Batch processing tab for CSV upload."""
    st.header("Batch Processing")

    available_models = get_available_models()

    if available_models:
        model_choice = st.selectbox(
            "Select Model",
            options=["No model (blank)"] + list(available_models.keys()),
            index=0,
            key="batch_model",
        )
    else:
        model_choice = "No model (blank)"
        st.info("No trained models found. Train a model in the Training tab.")

    model_path = None
    if model_choice != "No model (blank)" and model_choice in available_models:
        model_path = str(available_models[model_choice])

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(df.head())

        text_column = st.selectbox(
            "Select text column",
            options=df.columns.tolist(),
        )

        if st.button("Process", type="primary"):
            ner_model = load_ner_model(model_path)
            classifier = PredictionClassifier(ner_model)

            texts = df[text_column].tolist()

            progress = st.progress(0)
            results = []

            for i, text in enumerate(texts):
                result = classifier.classify(str(text))
                results.append({
                    "text": text,
                    "prediction_type": result.prediction_type.value,
                    "source": result.components.source or "",
                    "target": result.components.target or "",
                    "date": result.components.date or "",
                    "outcome": result.components.outcome or "",
                    "components_found": result.components.component_count,
                    "confidence": result.confidence_score,
                })
                progress.progress((i + 1) / len(texts))

            results_df = pd.DataFrame(results)
            st.success(f"Processed {len(results)} texts")

            st.subheader("Results Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                full_count = sum(1 for r in results if r["prediction_type"] == "full")
                st.metric("Full Predictions", full_count)
            with col2:
                partial_count = sum(1 for r in results if r["prediction_type"] == "partial")
                st.metric("Partial Predictions", partial_count)
            with col3:
                none_count = sum(1 for r in results if r["prediction_type"] == "not_a_prediction")
                st.metric("Not Predictions", none_count)

            st.subheader("Processed Data")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "prediction_results.csv",
                "text/csv",
                key="download-csv",
            )


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Prediction NER Demo",
        page_icon="🔮",
        layout="wide",
    )

    init_session_state()
    setup_logging()

    st.title("Prediction NER Demo")
    st.markdown(
        "Detect and classify prediction components in text using SpaCy NER models."
    )

    st.sidebar.header("About")
    st.sidebar.markdown(
        """
        **Prediction Components:**
        - **P_SOURCE** (p_s): Who makes the prediction
        - **P_TARGET** (p_t): What is being predicted about
        - **P_DATE** (p_d): When it should come true
        - **P_OUTCOME** (p_o): The predicted outcome

        **Classification:**
        - **Full**: All 4 components present
        - **Partial**: Some components present
        - **Not a Prediction**: No components found
        """
    )

    st.sidebar.header("Logs")
    if st.sidebar.button("Clear Logs"):
        clear_logs()
        st.rerun()

    with st.sidebar.expander("View Logs", expanded=False):
        if st.session_state.logs:
            log_text = "\n".join(reversed(st.session_state.logs[-100:]))
            st.code(log_text, language="text")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Inference",
        "Training",
        "Evaluation",
        "Batch Processing",
    ])

    with tab1:
        inference_tab()

    with tab2:
        training_tab()

    with tab3:
        evaluation_tab()

    with tab4:
        batch_processing_tab()


if __name__ == "__main__":
    main()
