# Predictions

A project to recognize and assess prediction data (text, numerical, audio, visual). The goal is to provide an analysis of how accurate predictions are.

## Table of Contents

- [Definitions](#definition)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## File Structure

```plaintext
├── misc                        # Contains random pieces of unfinished code.
├── prediction_classification   # Contains the pipeline to classify if a sentence is a prediction or not. 
├── prediction_correctness      # Contains the pipeline to assess how similar a prediction is to an actual outcome.
├── classification_models.py    # Contains the models to classify if a sentence is a prediction or not. 
├── clean_predictions.py        # Contains the code to clean our data.
├── data_processing.py          # Contains the code to manipulate our data.
├── feature_extraction.py       # Contains the code to extract features from predictions.
├── log_files.py                # Contains the code to produce a log file.
├── requirements.py             # Contains the requiremmts to run code in project.
├── text_generation_models.py   # Contains the LLMs to generate our data.
└── README.md                   # Project documentation
```

## Installation
> Use the package manager you prefer.

1. Install the [uv package manager]([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/getting-started/installation/#pypi)). For macOS, you can use `brew install uv`,
2. OPTIONAL: Create a project with `uv init .` that'll default to name of directory. It may need to be repository name `predictions`, so you could try `uv init predictions`
   - If you already see a `.toml` file, you should be able to skip.
4. Create virtual environment with `uv venv` or `uv venv <name>` (`uv venv .venv_predictions`)
5. Activate virtual environment with `source .venv/bin/activate` or `source .<name>/bin/activate` (`source .venv_predictions/bin/activate`)
6. Install requirements with `uv pip install -r pyproject.toml`
7. Create a `.env` file
     - Create a [Groq Cloud](https://console.groq.com/) API key -- `GROQ_CLOUD_API_KEY = "djb"`
     - Create a [NaviGator](https://api.ai.it.ufl.edu/) API key -- `NAVI_GATOR_API_KEY = "djb2"`
