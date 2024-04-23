## Local Development
Create a conda virtual environment and activate it:

```bash
conda create --name <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

Run Streamlit App (if you have model weights, else train model first)
```bash
streamlit run app.py
```

If you wish to train the NER model
```bash
python3 app/services/model.py
```