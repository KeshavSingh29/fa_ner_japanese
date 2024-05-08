FROM python:3.10

WORKDIR /usr/src/app

COPY . .

EXPOSE 8080

RUN pip install -r requirements.txt --no-cache-dir && export PYTHONPATH="$PWD"

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
