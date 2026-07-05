FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch numpy pandas pyarrow && \
    pip install --no-cache-dir scikit-learn scipy
ENTRYPOINT ["python3", "cli.py"]
