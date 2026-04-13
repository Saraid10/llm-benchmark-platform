FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "gradio>=4.0,<5.0" \
    duckdb \
    pandas \
    numpy \
    pydantic \
    plotly \
    psutil \
    rich \
    python-dotenv

COPY . .
RUN mkdir -p data/raw data/versions
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "app.py"]
