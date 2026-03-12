FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY frontend/requirements.txt /app/frontend/requirements.txt

RUN pip install --upgrade pip && pip install -r /app/frontend/requirements.txt

COPY frontend /app/frontend
COPY src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
