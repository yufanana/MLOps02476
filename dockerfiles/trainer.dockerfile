# Base image
FROM python:3.10-slim

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_yf/ mlops_yf/
COPY data/ data/

# Set working directory
WORKDIR /
RUN mkdir -p models/
RUN pip install .

ENTRYPOINT ["python", "-u", "mlops_yf/train_model.py"]
