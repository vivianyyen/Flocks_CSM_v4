import streamlit as st
from dotenv import load_dotenv
from utils.supabase_adapter import get_supabase_data
from utils.ai_classifier_v2 import ai_classifier_v2

load_dotenv()

st.set_page_config(
    page_title="AI Incident Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create the data fetcher for Supabase
get_data = get_supabase_data('cyber_news')

# Run the classifier page
ai_classifie_v2r(get_data)
