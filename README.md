# DL Final Project

### Set up virtual environment
    python3 -m venv venv
    . venv/bin/activate

### Install requirements
    pip install -r requirements.txt

### Download spacy vocab
    python -m spacy download en_core_web_sm

### Run training loop
    python main.py

### Run Evaluation
    python eval.py <model_name>

Where `<model_name> references one of our saved models. For example:
    python eval.py ResNextCNNtoRNNSingleLayer

This will output BLEU, GLEU, METEOR scores, as well as number of parameters.
For example:
    BLEU SCORE: 2.0062791424886875e-78
    GLEU SCORE: 0.016666666666666666
    METEOR SCORE: 0.3357603569554028
    TOTAL PARAMS: 92422528
    TOTAL TRAINABLE PARAMS 5680192