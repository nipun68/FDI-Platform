FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the Interaction Engine (3 Models) during build
RUN python src/data_ingestion/generate_synthetic_data.py
RUN python src/ml_models/train_striker_model.py
RUN python src/ml_models/train_goalkeeper_model.py
RUN python src/ml_models/train_outcome_model.py

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]