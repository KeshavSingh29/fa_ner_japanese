## Model Description

This model is a fine-tuned version of the `tohoku-nlp/bert-base-japanese-v3`, specifically optimized for Named Entity Recognition (NER) tasks. 
It is fine-tuned using a Japanese named entity extraction dataset derived from Wikipedia, which was developed and made publicly available by Stockmark Inc. ([NER Wikipedia Dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset)).

## Intended Use

This model is intended for use in tasks that require the identification and categorization of named entities within Japanese text. 
It is suitable for various applications in natural language processing where understanding the specific names of people, organizations, locations, etc., is crucial.

## How to Use

### Download Model Weights/Files

Get the model weights from huggingface and save it to `ner_model` directory. 
*In future I will make this process more straightforward*
```bash
cd ner_model
wget https://huggingface.co/knosing/japanese_ner_model/resolve/main/model.safetensors
wget https://huggingface.co/knosing/japanese_ner_model/resolve/main/training_args.bin
wget https://huggingface.co/knosing/japanese_ner_model/resolve/main/config.json
```


### Local Development

### Conda

Create a conda virtual environment and activate it:

```bash
conda create --name ner_fa python=3.10
conda activate ner_fa
pip install -r requirements.txt --no-cache-dir
export PYTHONPATH="$PWD"
```

Run Streamlit App (if you have model weights, else train model first)
```bash
streamlit run app.py
```

#### Training

If you wish to `train` the NER model
```bash
python3 app/services/model.py
```

### Docker

Train the NER model and save it to `ner_model` folder. 
It should contain the following files: 
- config.json
- model.safetensors
- training_args.bin

Create image (will take about 5min)
```bash
docker build . -t ner
```

Run
```bash
docker run -p 8080:8080 ner
```

### Result
The model has been evaluated on various entity types to assess its precision, recall, F1 score, and overall accuracy. Below is the detailed performance breakdown by entity type:

#### Overall Metrics

- **Overall Precision:** 0.8379
- **Overall Recall:** 0.8477
- **Overall F1 Score:** 0.8428
- **Overall Accuracy:** 0.9684

#### Performance by Entity Type

- **Other Organization Names (`の他の組織名`):**
  - **Precision:** 0.71875
  - **Recall:** 0.69
  - **F1 Score:** 0.7041
  - **Sample Count:** 100

- **Event Names (`ベント名`):**
  - **Precision:** 0.85
  - **Recall:** 0.8586
  - **F1 Score:** 0.8543
  - **Sample Count:** 99

- **Personal Names (`人名`):**
  - **Precision:** 0.8171
  - **Recall:** 0.8664
  - **F1 Score:** 0.8410
  - **Sample Count:** 232

- **Generic Names (`名`):**
  - **Precision:** 0.8986
  - **Recall:** 0.9376
  - **F1 Score:** 0.9177
  - **Sample Count:** 529

- **Product Names (`品名`):**
  - **Precision:** 0.6522
  - **Recall:** 0.5906
  - **F1 Score:** 0.6198
  - **Sample Count:** 127

- **Government Organization Names (`治的組織名`):**
  - **Precision:** 0.9160
  - **Recall:** 0.8276
  - **F1 Score:** 0.8696
  - **Sample Count:** 145

- **Facility Names (`設名`):**
  - **Precision:** 0.7905
  - **Recall:** 0.8357
  - **F1 Score:** 0.8125
  - **Sample Count:** 140


### Notes


Have any issues? Please feel free to contact me.