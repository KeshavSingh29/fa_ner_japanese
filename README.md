## Local Development
Create a conda virtual environment and activate it:

```bash
conda create --name <env_name> python=3.10 --file requirements.txt
conda activate <env_name>
```

Run Streamlit App
```bash
streamlit run app.py
```

If you wish to train the NER model again
```bash
python3 services/model.py
```