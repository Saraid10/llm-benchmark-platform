FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "jinja2==3.1.4" \
    "huggingface_hub==0.23.4" \
    "gradio==4.26.0" \
    "starlette==0.27.0" \
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
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "app.py"]
