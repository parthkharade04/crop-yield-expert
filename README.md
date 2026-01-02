# 🌾 Smart Crop Yield Prediction System

**An End-to-End Machine Learning & GenAI Project**

This project is a modern, full-stack data science application that predicts crop yields based on environmental factors. It demonstrates a complete pipeline from raw data engineering to a production-ready Web Dashboard integrated with Generative AI.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GenAI](https://img.shields.io/badge/GenAI-Groq%20Llama3-blueviolet?style=for-the-badge)

## 🚀 Features

### 1. 🔮 Precision Yield Prediction
- Uses an **Ensemble Model (Random Forest & XGBoost)** trained on historical data.
- Considers **Actual Seasonal Rainfall** (calculated from raw monthly logs), Soil pH, and location.
- Provides scientific yield estimates (Tonnes/Hectare).

### 2. 🤖 AI Agronomist (GenAI)
- Integrated **Llama 3 (via Groq API)** to act as an expert agricultural consultant.
- **Context-Aware:** The AI knows your specific prediction details (Crop, District, Rainfall) and gives tailored advice on fertilizers and farming techniques.

### 3. 📈 Data Analytics
- Interactive visualizations of historical trends.
- Analyzes the impact of rainfall patterns on different crops.

## 🛠️ Tech Stack
- **Data Engineering:** Python, Pandas (Complex joins, pivot tables, seasonal aggregations).
- **Machine Learning:** Scikit-Learn, XGBoost (Pipeline, Imputation, One-Hot Encoding).
- **App Framework:** Streamlit (Interactive Web Dashboard).
- **LLM Integration:** Groq API (Llama 3).

## 📂 Project Structure
```text
Crop Yield/
├── data/
│   ├── raw/               # Raw weather & production logs
│   └── processed/         # Engineered Master Dataset
├── models/                # Saved ML Models (.pkl)
├── src/
│   ├── data_ingestion.py  # ETL Pipeline (Raw -> Master)
│   └── training.py        # ML Training Pipeline
├── streamlit_app.py       # Main Web Application
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## 💻 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/crop-yield-expert.git
    cd crop-yield-expert
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **Configure AI (Optional)**
    - To use the AI Agronomist, you need a Groq API Key.
    - Enter it in the code or set it up in `.streamlit/secrets.toml`.

## ☁️ Deployment (Streamlit Cloud)

1.  Push this code to a **GitHub Repository**.
2.  Log in to [Streamlit Cloud](https://share.streamlit.io/).
3.  Click **New App** and select your repository.
4.  **Important:** Go to `App Settings -> Secrets` and add your API Key:
    ```toml
    GROQ_API_KEY = "your_groq_api_key_here"
    ```
5.  Click **Deploy**!

---
*Built by Parth Kharade*
