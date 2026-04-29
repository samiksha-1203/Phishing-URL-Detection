# Phishing URL Detection

A phishing URL detection project using the PhiUSIIL dataset, with a Streamlit app for live predictions and a training script for model generation.

## Project Structure

```text
Phishing/
|-- src/
|   |-- app.py
|   `-- phishing_detection.py
|-- data/
|   `-- raw/
|       `-- PhiUSIIL_Phishing_URL_Dataset.csv
|-- artifacts/
|   |-- graphs/
|   `-- models/
|-- docs/
`-- .gitignore
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Train and generate graphs/models:

```bash
python src/phishing_detection.py
```

Start Streamlit app:

```bash
streamlit run src/app.py
```

## Notes

- Dataset path expected at `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`.
- Trained model files are saved in `artifacts/models/`.
- Output charts are saved in `artifacts/graphs/`.
