## Local Development
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

If you wish to train the NER model
```bash
python3 app/services/model.py
```

Have any issues? 
Please feel free to contact me. 