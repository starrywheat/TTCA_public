FROM python:3.10-slim

WORKDIR /ttca

RUN apt-get update && apt-get install -y git

RUN pip install --upgrade pip

COPY ./requirements.txt /ttca

RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install --upgrade openai

# RUN python -m spacy download en_core_web_sm

COPY . /ttca

ENV PYTHONPATH "{PYTHONPATH}:/ttca"

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "server.port=8501", "--server.address=0.0.0.0"]
