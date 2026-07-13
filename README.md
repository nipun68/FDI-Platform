<p align="center">
  <a href="https://github.com/nipun68/fdi-platform">
    <img src="public/og-image.png" alt="DuelTactix AI — Sports AI Decision Intelligence" width="100%" />
  </a>
</p>

<h1 align="center">DuelTactix AI</h1>

<p align="center">
  <strong>An end-to-end Sports AI Decision Intelligence platform modeling the psychological duel between striker and goalkeeper. By combining structured match data, Computer Vision biomechanics, Deep Learning embeddings, and Generative AI, the system provides explainable predictions, simulates thousands of match outcomes, and recommends optimal strategies.</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-modules--ecosystem">Modules & Ecosystem</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-project-structure">Project Structure</a> ·
  <a href="#-development--building">Development & Setup</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?color=6C5CE7" alt="python version" />
  <img src="https://img.shields.io/badge/FastAPI-v0.109.2-8A2BE2?color=6C5CE7" alt="fastapi version" />
  <img src="https://img.shields.io/badge/react-18.3.1-61DAFB?color=6C5CE7" alt="react version" />
  <img src="https://img.shields.io/badge/catboost-1.2.5-orange?color=6C5CE7" alt="catboost version" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/badge/simulations-1000%2B-blue" alt="1000+ Simulations" />
</p>

<br/>

## <img src="./public/readme-assets/overview.svg" width="22" height="22" align="center" alt="" />&nbsp; Overview

DuelTactix AI (FDI Platform) is a state-of-the-art Sports AI Decision Intelligence engine. Traditional football analytics treat the striker and goalkeeper in isolation, providing static, black-box predictions that coaches do not trust. 

Our solution is a **multi-modal, interacting-agent engine** modeling the striker and goalkeeper as independent agents. By combining structured match data, computer vision biomechanics (pose estimation), deep learning entity embeddings, and generative AI, the system calculates the outcome of their specific duel, explains the predictions via SHAP, and simulates live shootout momentum in real-time.

All models are trained in multiple modular structures:
*   **Striker Agent Model**: CatBoost model predicting the striker's target zone choice and success probability.
*   **Goalkeeper Agent Model**: CatBoost model predicting goalkeeper dive direction based on historical, biographical, and spatial features.
*   **Outcome Interaction Model**: Models the actual physical/psychological duel from the combination of action vectors, generating explainable (SHAP) probabilities.
*   **Deep Learning Embeddings (LSTM/Player Embeddings)**: Deep learning representation vectors modeling temporal patterns over time.



## <img src="./public/readme-assets/packages.svg" width="22" height="22" align="center" alt="" />&nbsp; Modules & Ecosystem

| Logo | Module | Tech Stack | Description | Links |
| :---: | :--- | :--- | :--- | :--- |
| <img src="https://cdn.simpleicons.org/python/3776AB" alt="Python logo" width="30"> | **`fdi-api`** | FastAPI · SlowAPI · PyJWT | Secure REST backend with rate limiting and JWT auth. | [Source](./src/api) |
| <img src="https://cdn.simpleicons.org/react/61DAFB" alt="React logo" width="30"> | **`fdi-frontend`** | React 18 · Vite · Recharts | Modern glassmorphism coach dashboard with data visualizations. | [Source](./frontend) |
| <img src="https://cdn.simpleicons.org/opencv/5C3EE8" alt="OpenCV logo" width="30"> | **`fdi-cv`** | OpenCV · MediaPipe Pose | Video pose analysis engine extracting body lean angle and run-up speed. | [Source](./src/cv_module) |
| <img src="https://cdn.simpleicons.org/pytorch/EE4C2C" alt="PyTorch logo" width="30"> | **`fdi-ml`** | CatBoost · PyTorch · SHAP | Core models, interaction engine, and player embeddings. | [Source](./src/ml_models) |
| <img src="https://cdn.simpleicons.org/numpy/013243" alt="NumPy logo" width="30"> | **`fdi-simulation`** | NumPy · Scikit-Learn | Vectorized Monte Carlo engine for simulating shootout trajectories. | [Source](./src/simulation) |
| <img src="https://cdn.simpleicons.org/terraform/844FBA" alt="Terraform logo" width="30"> | **`fdi-infrastructure`** | Terraform · Kubernetes · Prometheus | Declarative IaC, K8s manifests, and Prometheus/Grafana monitoring. | [Source](./infrastructure) |



## <img src="./public/readme-assets/quick-start.svg" width="22" height="22" align="center" alt="" />&nbsp; Quick Start

### 1. Authenticate and Fetch Token

```python
import requests

url = "http://localhost:8000/token"
response = requests.get(url).json()
access_token = response["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}
```

### 2. Multi-Modal Duel Prediction (FastAPI)

Provide features including Computer Vision biomechanics (`cv_body_lean_angle`, `cv_run_up_speed`) to predict goalkeeper actions and outcome probabilities with SHAP explanations:

```python
payload = {
    "player_id": 42,
    "player_penalty_conversion_rate": 0.85,
    "career_penalty_attempts": 20,
    "recent_penalty_form": 0.90,
    "is_shootout": 1,
    "match_stage": "Final",
    "minute_of_match": 120,
    "fatigue_index": 0.65,
    "run_up_style": "Slow-Fast",
    "pause_before_shot": 1,
    "cv_body_lean_angle": 18.5,
    "cv_run_up_speed": 8.7,
    "goalkeeper_id": 1,
    "goalkeeper_save_rate": 0.32,
    "goalkeeper_diving_bias": "Left",
    "keeper_experience_level": 4,
    "shooter_vs_keeper_history": 2,
    "psychological_advantage_index": 7,
    "shot_direction": "Bottom Left"
}

res = requests.post("http://localhost:8000/predict/agents", json=payload, headers=headers).json()
print(f"Goal Probability: {res['interaction_outcome']['probability_as_percentage']}")
print(f"Top Driving Factor: {res['ai_explanation'][0]['feature']}")
```

### 3. Analyze Penalty Video Biomechanics (OpenCV/MediaPipe)

Upload video to automatically run pose estimation and extract biomechanics:

```python
files = {'file': open('penalty_runup.mp4', 'rb')}
res = requests.post("http://localhost:8000/analyze/video", files=files, headers=headers).json()
print(f"Extracted Body Lean: {res['cv_body_lean_angle']}°")
print(f"Extracted Run-Up Speed: {res['cv_run_up_speed']} m/s")
```

### 4. Generate GenAI Tactical Coach Report (Groq Llama 3.3)

Convert AI predictions and SHAP factors into a professional human-like tactical description:

```python
report_payload = {
    "xg": 0.825,
    "cv_lean": 18.5,
    "fatigue": 0.65,
    "reasons": [
        {"feature": "cv_body_lean_angle", "direction": "Increased", "impact": 0.12},
        {"feature": "fatigue_index", "direction": "Decreased", "impact": 0.08}
    ]
}

res = requests.post("http://localhost:8000/generate/report", json=report_payload, headers=headers).json()
print(res["tactical_report"])
```



## <img src="./public/readme-assets/structure.svg" width="22" height="22" align="center" alt="" />&nbsp; Project Structure

This monorepo houses the core machine learning models, video analytics, backend services, client applications, and declarative infrastructure code.

```
├── src/
│   ├── api/                 # FastAPI backend (JWT, Rate Limiting, Groq GenAI, Video Upload)
│   │   └── main.py          # Entrypoint & endpoint routes for PIS²
│   ├── cv_module/           # Biomechanical feature extractor (MediaPipe Pose)
│   │   └── pose_estimator.py
│   ├── data_ingestion/      # StatsBomb integration & synthetic generator
│   │   ├── download_statsbomb.py
│   │   └── generate_synthetic_data.py
│   ├── feature_engineering/ # Spatial geometry & feature pipeline
│   │   └── create_features.py
│   ├── ml_models/           # CatBoost models & PyTorch temporal embedders
│   │   ├── explain_model.py
│   │   ├── train_interaction_models.py
│   │   └── lstm_temporal.py
│   └── simulation/          # Shootout Monte Carlo simulator
│       └── simulate_shootout.py
├── frontend/                # React.js coach UI & Recharts visualization
├── infrastructure/          # Orchestration and configuration manifests
│   ├── k8s/                 # Kubernetes deployment specs
│   ├── terraform/           # IaC config (main.tf)
│   └── monitoring/          # Prometheus monitoring rules
├── tests/                   # Pytest automated test suites
├── Dockerfile               # Production container specs
└── requirements.txt         # Root Python requirements
```



## <img src="./public/readme-assets/design.svg" width="22" height="22" align="center" alt="" />&nbsp; Design System

The DuelTactix coach dashboard follows a strict, high-fidelity **Glassmorphic design system** styled with custom CSS variables.
*   **Vibrant Theme**: Slate/Jet dark interface paired with glowing neon accents (violet `#6C5CE7` and cyan `#00E5FF`).
*   **Visualizations**: Custom Recharts layout using Radial Gauges and Diverging Bar Charts to make SHAP coefficients human-readable for coaches.
*   **Layout Rules**: Highly responsive CSS grid layout suited for tablet-sized pitch-side tablets or full desktop screens.



## <img src="./public/readme-assets/development.svg" width="22" height="22" align="center" alt="" />&nbsp; Development & Building

### Prerequisites
*   Python 3.11+
*   Node.js v18+ & npm

### Local Installation

1.  **Clone the repository and install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Dataset & Train Models**:
    Generate the 50,000-row synthetic database and train all three CatBoost agents (Striker, GK, Outcome):
    ```bash
    python src/data_ingestion/generate_synthetic_data.py
    python src/ml_models/train_striker_model.py
    python src/ml_models/train_goalkeeper_model.py
    python src/ml_models/train_outcome_model.py
    ```

3.  **Start the REST API Backend**:
    ```bash
    uvicorn src.api.main:app --reload
    ```

4.  **Launch the React Dashboard**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```



## <img src="./public/readme-assets/contributing.svg" width="22" height="22" align="center" alt="" />&nbsp; Contributing

Contributions to improve model weights, biomechanical formulas, or dashboard layouts are welcome. Please refer to standard Git workflows to submit a Pull Request.



## <img src="./public/readme-assets/star-history.svg" width="22" height="22" align="center" alt="" />&nbsp; Our Stargazers

[![Stargazers](https://github-readme-stargazers-five.vercel.app/api/card?username=nipun68&repo=fdi-platform&width=1200&theme=Snowy+Minimal+%28Solid%29&rows=4&cols=8&wreath=true&radius=7&card_border=false&blur=1.5&badges=fire&badge_density=0.45&bg_style=solid&v=1)](https://github.com/nipun68/fdi-platform/stargazers)

<br/>



## <img src="./public/readme-assets/license.svg" width="22" height="22" align="center" alt="" />&nbsp; Credits

DuelTactix AI makes use of open datasets and foundational packages:
*   [StatsBomb Open Data](https://github.com/statsbomb/open-data) for standard penalty historical records.
*   [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) by Google for real-time body tracking.
*   [CatBoost](https://catboost.ai/) for high-accuracy gradient boosting classification.



## <img src="./public/readme-assets/license.svg" width="22" height="22" align="center" alt="" />&nbsp; License

MIT License - Copyright (c) 2026 [Nipun Kalra](https://github.com/nipun68).