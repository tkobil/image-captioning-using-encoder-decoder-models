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

### Choosing the model type
We provide both LSTM and GRU based models. Please see `model.py` and `model_gru.py` respectively.

### Evaluating Models
Please see the `results/` directory for epoch loss data in csv files. We've included 
`.ipynb` notebooks for each model to analyze various metrics and run inference.

- `resnext_gru_eval_3_layer.ipynb`
- `resnext_lstm_eval_single_layer.ipynb`
- `resnext_lstm_eval_3_layer.ipynb`
- `resnect_gru_eval_single_layer.ipynb`

`eval.ipynb` is provided as a reference template notebook for evaluating a model.

NOTE: `.pt` model files/weights are available upon request. We have NOT included
them in this repository due to the size of the model files.
