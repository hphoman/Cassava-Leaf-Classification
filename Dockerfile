FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch==2.14.0 \
    torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu126

COPY src/ src/
COPY models/ models/

RUN mkdir -p results figures

CMD ["python", "-m", "src.run_analysis"]
