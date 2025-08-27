# ✈️ Airline Delay Prediction  

A predictive machine learning pipeline to analyze and forecast flight delays in the United States.  
This project leverages real airline delay data, trains a machine learning model, and serves predictions through an **interactive Streamlit dashboard**.  

---

## 📊 Workflow  

```mermaid
flowchart TD
    A[Start] --> B[Preprocess Dataset]
    B --> C[Exploratory Data Analysis]
    C --> D[Model Training - Random Forest]
    D --> E{Is Flight Delayed?}
    E -- Yes --> F[Predict Delay Minutes]
    E -- No --> G[On Time]
    F --> H[Dashboard Visualization]
    G --> H
    H --> I[End]
📖 Abstract
Air travel delays are one of the most common challenges faced by airlines and passengers.
Delays lead to financial loss, missed connections, and passenger dissatisfaction.

This project develops a data-driven delay prediction system that:

Identifies the major causes of delays (weather, carrier, security, NAS)

Builds a regression model to predict delay duration

Provides an interactive dashboard for visualization and prediction

With this tool, decision-makers can analyze historical patterns and make proactive operational choices.

📂 Repository Structure
bash
Copy code
├── .github/              # GitHub workflows
├── data/                 # Dataset
│   └── Airline_Delay_Cause.csv
├── notebooks/            # Jupyter notebooks (EDA + experiments)
├── src/                  # Source code
│   └── train_airline_delay.py
├── configs/              # Model configs
├── app.py                # Streamlit dashboard
├── requirements.txt      # Dependencies
└── README.md             # Project docs
⚙️ Installation
Clone the repository

bash
Copy code
git clone https://github.com/G0wth9m/airline-delay-prediction.git
cd airline-delay-prediction
Set up virtual environment

bash
Copy code
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Mac/Linux
Install dependencies

bash
Copy code
pip install -r requirements.txt
🚀 Usage
Train Model
bash
Copy code
python -m src.train_airline_delay --csv data/Airline_Delay_Cause.csv
Run Dashboard
bash
Copy code
streamlit run app.py
Open your browser at 👉 http://localhost:8501

🌟 Features
📌 Data cleaning & preprocessing

📊 Exploratory Data Analysis (EDA)

🌲 Random Forest model for regression

🎛 Interactive dashboard (Streamlit)

📈 Visualizations for delay causes & predictions

🛠️ Tech Stack
Python (Pandas, NumPy, Scikit-learn)

Matplotlib / Seaborn for visualization

Streamlit for the dashboard

Git + GitHub for version control

📷 Dashboard Preview
