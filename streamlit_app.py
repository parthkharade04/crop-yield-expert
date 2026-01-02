import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq

# Page Config
st.set_page_config(page_title="Smart Crop Yield Engine", page_icon="🌾", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; color: #333333; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model_path = 'models/crop_yield_model.pkl'
    data_path = 'data/processed/master_dataset.csv'
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        return joblib.load(model_path), pd.read_csv(data_path)
    return None, None

model, df = load_resources()

if model is None:
    st.error("Resources not found. Please run training first.")
    st.stop()

# --- Main Layout ---
st.title("🌾 Smart Crop Yield Prediction & Analytics")

# AI KEY Setup
# Try to get key from secrets (Cloud), else use local backup or empty
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Fallback for local testing if not set in secrets.toml
    # Make sure to remove real key before committing if not using secrets!
    GROQ_API_KEY = "" # Key removed for security. Set in st.secrets for Cloud.

tabs = st.tabs(["🔮 Prediction Dashboard", "🤖 AI Agronomist", "📈 Trends & Insights"])

# --- TAB 1: Prediction ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Farm Parameters")
        districts = sorted(df['District_Name'].unique())
        seasons = sorted(df['Season'].unique())
        crops = sorted(df['Crop'].unique())
        
        sel_district = st.selectbox("District", districts)
        sel_season = st.selectbox("Season", seasons)
        sel_crop = st.selectbox("Crop", crops)
        
        area = st.number_input("Area (Ha)", 1.0, 100.0, 10.0)
        max_rain = float(df['Actual_Rainfall'].max())
        rainfall = st.slider("Rainfall (mm)", 0.0, max_rain, 800.0)
        ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
        
        input_df = pd.DataFrame([{
            'District_Name': sel_district, 'Season': sel_season, 'Crop': sel_crop,
            'Area': area, 'Actual_Rainfall': rainfall, 
            'pH_min': ph, 'pH_max': ph, 'pH_min_Range': 5.5, 'pH_max_Range': 8.5
        }])
        
        if st.button("Predict"):
            pred = model.predict(input_df)[0]
            st.session_state['last_pred'] = pred
            st.session_state['last_area'] = area
            # Save context for AI
            st.session_state['context'] = {
                'district': sel_district, 'crop': sel_crop, 'season': sel_season,
                'rainfall': rainfall, 'ph': ph, 'yield': pred/area
            }

    with col2:
        st.subheader("Prediction Result")
        if 'last_pred' in st.session_state:
            val = st.session_state['last_pred']
            yld = val / st.session_state['last_area']
            
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='metric-card'><h3>Production</h3><h2 style='color:green'>{val:,.1f} Tonnes</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h3>Yield</h3><h2 style='color:blue'>{yld:,.1f} Tonnes/Ha</h2></div>", unsafe_allow_html=True)
            
            st.success("Analysis: Prediction generated successfully based on modern Ensemble Learning.")
        else:
            st.info("Enter details and click Predict.")

# --- TAB 2: AI Agronomist ---
with tabs[1]:
    st.header("🤖 AI Agronomist Assistant")
    
    if 'context' in st.session_state:
        ctx = st.session_state['context']
        st.info(f"Context: Analyzing {ctx['crop']} in {ctx['district']} (Rainfall: {ctx['rainfall']}mm, pH: {ctx['ph']}). Predicted Yield: {ctx['yield']:.2f} T/Ha")
        
        user_query = st.text_input("Ask a question about your crop, soil, or fertilizers:", 
                                  placeholder="e.g., How can I improve my yield with this low rainfall?")
        
        if st.button("Ask AI"):
            if not GROQ_API_KEY:
                st.error("Please configure Groq API Key in code.")
            else:
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # Prompt Engineering
                    prompt = f"""
                    You are an expert Agronomist and Agricultural Scientist.
                    Context:
                    - Crop: {ctx['crop']}
                    - District: {ctx['district']} (Maharashtra, India)
                    - Season: {ctx['season']}
                    - Soil pH: {ctx['ph']}
                    - Rainfall: {ctx['rainfall']} mm
                    - Predicted Yield: {ctx['yield']:.2f} Tonnes/Hectare
                    
                    User Question: {user_query}
                    
                    Provide a concise, practical, and scientific answer. Suggest fertilizers or farming techniques if asked.
                    """
                    
                    with st.spinner("Consulting AI Expert..."):
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a helpful agricultural expert."},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama3-8b-8192",
                        )
                        response = chat_completion.choices[0].message.content
                        st.markdown("### 💡 AI Advice:")
                        st.write(response)
                        
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.warning("⚠️ Please go to the 'Prediction Dashboard' tab and make a prediction first to set the context.")

# --- TAB 3: Trends ---
with tabs[2]:
    st.header("📈 Historical Trends")
    trend_crop = st.selectbox("Analyze Crop", sorted(df['Crop'].unique()), key='trend')
    trend_data = df[df['Crop'] == trend_crop]
    
    fig2 = px.scatter(
        trend_data, x="Actual_Rainfall", y="Production", 
        color="Season", size="Area", 
        hover_data=["District_Name", "Crop_Year"],
        title=f"Rainfall vs Production Analysis: {trend_crop}"
    )
    st.plotly_chart(fig2, use_container_width=True)
