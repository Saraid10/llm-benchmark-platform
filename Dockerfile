FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pin huggingface_hub FIRST before gradio pulls in a newer version
RUN pip install --no-cache-dir huggingface_hub==0.23.4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/versions

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "src/ui/app.py"]
