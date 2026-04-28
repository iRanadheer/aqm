# Wind Opposition Detection

A project with the CDL at Brown to build a taxonomy and classifier for wind energy opposition narratives and claims. The classifier uses Claude with chain-of-thought prompting to identify opposition frames in text.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

## Classification

The core classifier lives in `classifier.py`. It sends text to Claude along with a detailed taxonomy of opposition narratives and claims, and returns structured labels.

```python
from classifier import WindOppositionClassifier
import os

classifier = WindOppositionClassifier(os.getenv("ANTHROPIC_API_KEY"))

result = classifier.classify("Wind turbines kill birds and cause noise pollution.", use_cot=True)
print(result["narratives"])  # e.g., N_1 (Wildlife/environmental harm)
print(result["claims"])      # e.g., C_1_1 (Bird/bat harm)
```

Batch classification is also supported:

```python
results = classifier.classify_batch(texts, use_cot=True)
```

## Taxonomy

The taxonomy defines 8 opposition narratives and 39+ specific claims, organized hierarchically. Narratives are broad frames (e.g., "Wind energy harms wildlife") and claims are specific arguments within each frame (e.g., "Bird/bat mortality").

The full taxonomy is defined in `codebooks.py`. The system prompt that instructs the classifier is in `prompts.py`.

## Performance Evaluation

`performance.py` provides tools for evaluating classifier accuracy against labeled data. It computes per-class precision, recall, and F1 for both narratives and claims, and supports comparison across different model configurations.

## Error Analysis

Three scripts support error analysis:

- `analyze_errors.py` -- detailed analysis of misclassifications with exportable reports
- `explore_errors.py` -- interactive exploration of prediction errors
- `quick_error_summary.py` -- concise summary of where the classifier struggles

All three build on utilities from `performance.py`.

## Project Structure

```
classifier.py           # Main classifier (Claude + chain-of-thought)
prompts.py              # System prompt for classification
codebooks.py            # Narrative and claim taxonomy definitions
performance.py          # Evaluation metrics and model comparison
analyze_errors.py       # Detailed error analysis
explore_errors.py       # Interactive error exploration
quick_error_summary.py  # Quick error summary
load_training_data.py   # Load and parse training examples
data/                   # Data files and older analysis artifacts
```

## Requirements

- Python 3.10+
- anthropic
- pandas
- scikit-learn
- python-dotenv

## License

MIT. See [LICENSE](LICENSE).
