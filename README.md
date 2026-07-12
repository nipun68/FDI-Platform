⚔️ DuelTactix AI (FDI Platform)
An end-to-end Sports AI Decision Intelligence platform that models the psychological duel between striker and goalkeeper. By combining structured match data, Computer Vision biomechanics, Deep Learning embeddings, and Generative AI, the system provides explainable predictions, simulates thousands of match outcomes, and recommends optimal strategies.

PythonFastAPICatBoostReactGroq

🧠 The Problem We Solved
Traditional football analytics treat the striker and goalkeeper in isolation, providing static, black-box predictions that coaches don't trust.

Our Solution: DuelTactix AI is a multi-modal, interacting-agent engine. We model the striker and GK as independent agents, calculate the outcome of their specific duel, explain the math via SHAP and LLMs, and track live shootout momentum in real-time.

🚀 Core Features (The 3 Killer Differentiators)
Computer Vision Video Upload: Coaches can upload a video of a penalty. The API uses OpenCV/MediaPipe to extract real biomechanical data (body_lean_angle, run_up_speed) and fuses it with tabular data for multi-modal predictions.
GenAI Tactical Coach: Powered by Llama 3.3 (via Groq). Takes the mathematical SHAP values and generates a human-like tactical scouting report (e.g., "Coach, based on the video, his body lean is 18°. Combined with fatigue, he has a 40% chance...").
Live Shootout Momentum Tracker: A dynamic Monte Carlo engine. As the coach clicks "Goal" or "Miss" on the UI, the API runs 1,000 new simulations starting from that exact live score state, updating win probabilities in real-time.
🏗 System Architecture
Data Layer: StatsBomb API integration + 50,000-row synthetic physics generator.
Machine Learning: 3-Model Interaction Engine (Striker Agent, GK Agent, Outcome Duel) using CatBoost. Deep Learning entity embeddings via PyTorch.
Computer Vision: MediaPipe Pose Estimation for biomechanical feature extraction.
Explainable AI (XAI): SHAP TreeExplainer integrated directly into the API payload, visualized via Recharts Diverging Bar Charts.
Simulation Engine: Vectorized Monte Carlo shootout simulator.
Backend: FastAPI, JWT Authentication, Rate Limiting (SlowAPI).
Frontend: React.js (Vite), Recharts (Radial Gauges), Glassmorphism UI.
DevOps & Cloud: Docker, Render.com (CI/CD), Terraform (IaC), Kubernetes manifests.
🛠 Tech Stack
Backend: Python, FastAPI, Uvicorn, OpenCV, MediaPipe
Machine Learning: CatBoost, Scikit-Learn, SHAP, PyTorch, NumPy
GenAI: Groq API (Llama 3.3 70B)
Frontend: React.js, Recharts, CSS3 (Glassmorphism UI)
DevOps: Docker, Render, GitHub Actions, Terraform, Kubernetes
⚙️ Local Setup Instructions
1. Clone the Repository
git clone https://github.com/nipun68/fdi-platform.gitcd fdi-platform
2. Backend & ML Setup
bash

python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt

# 1. Generate the 50,000 row dataset
python src/data_ingestion/generate_synthetic_data.py

# 2. Train the 3-Model Interaction Engine
python src/ml_models/train_striker_model.py
python src/ml_models/train_goalkeeper_model.py
python src/ml_models/train_outcome_model.py

# 3. Start the Secure FastAPI Backend
uvicorn src.api.main:app --reload
3. Frontend Setup
bash

cd frontend
npm install
npm run dev
Visit http://localhost:5173 to access the tactical dashboard. Click "Analyze Penalty Duel" to generate predictions, SHAP breakdowns, and GenAI reports.

📂 Project Structure
text

fdi-platform/
├── src/
│   ├── api/             # FastAPI backend (JWT, Rate Limiting, Groq GenAI, Video Upload)
│   ├── cv_module/       # MediaPipe Pose Estimation for video analysis
│   ├── data_ingestion/  # StatsBomb fetcher & 50k Synthetic Data Generator
│   ├── feature_engineering/ # Spatial math (angles, distances)
│   ├── ml_models/       # 3 Modular CatBoost training scripts & PyTorch Embedders
│   └── simulation/      # Vectorized Monte Carlo shootout engine
├── frontend/            # React.js tactical dashboard with Recharts
├── infrastructure/      # Terraform (AWS IaC), K8s manifests, Prometheus monitoring
├── tests/               # Pytest unit tests for pipeline integrity
├── Dockerfile           # Containerization config
└── requirements.txt     # Python dependencies
🔮 Future Roadmap
Real-time CV Tracking: Integrate live broadcast feeds instead of video uploads.
Reinforcement Learning: Train a defensive agent to optimize goalkeeper positioning.
Mobile App: React Native companion for sideline use.