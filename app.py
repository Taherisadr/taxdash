import streamlit as st
import pdfplumber
import json
import requests
import re
import html
from io import BytesIO

# --- Streamlit page setup ---
st.set_page_config(page_title="AI Tax Agent", layout="centered")
st.title("Welcome to GreenGrowth CPAs Tax Agent")

# === OpenRouter Configuration ===
OPENROUTER_API_KEY = st.secrets["api_keys"]["OPENROUTER_API_KEY"]
MODEL = "mistralai/mistral-7b-instruct"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# --- Initialize Chat Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "I'm your Tax Assistant. Ask me if you have any question."}
    ]

# === Text Refinement with LLM ===
def reformat_text_with_llm(raw_text: str) -> str:
    prompt = (
        "You are a formatting assistant. Your job is to clean and reformat the following text so that:\n"
        "- Words and numbers are properly spaced.\n"
        "- Punctuation is correct.\n"
        "- Paragraphs are separated clearly.\n"
        "- No information is added or removed.\n\n"
        "Fix this:\n\n" + raw_text
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You clean up model outputs by fixing spacing, punctuation, and formatting."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        with st.spinner("🧹 Cleaning up response..."):
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Cleanup LLM failed: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ Cleanup error: {str(e)}"

# === PDF Processing ===
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(texts)

# === Field Extraction with LLM ===
def extract_fields_from_text(text: str) -> dict:
    system_message = (
        "You are a helpful AI assistant that extracts structured tax information from W-2 form text. "
        "Return only a valid JSON object with the following keys:\n"
        "- Employee Name\n"
        "- Employer Name\n"
        "- Wages (Box 1)\n"
        "- Federal Income Tax Withheld (Box 2)\n"
        "- Social Security Wages (Box 3)\n"
        "- Filing Year\n\n"
        "Do not include any commentary or explanation—only valid JSON."
    )

    user_message = f"""
Here is the W-2 text:

{text[:1000]}

Extract the fields and respond only with JSON.
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    }

    with st.spinner("🧐 Extracting fields ..."):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30
        )

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        st.text_area("📝 Raw model output", reply, height=200)

        match = re.search(r'\{.*?\}', reply, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                parsed = json.loads(json_str)
                st.success("✅ Successfully parsed tax fields.")
                return parsed
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ JSON decoding failed: {str(e)}")
                return {}
        else:
            st.warning("⚠️ No JSON object found in model reply.")
            return {}
    else:
        st.error(f"❌ OpenRouter API failed: {response.status_code}\n{response.text}")
        return {}

# === Chat Interface (always at the bottom) ===
st.markdown("---")
st.subheader("🤖 Chat with GreenGrowth CPAs Tax Assistant")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
if user_input := st.chat_input("Ask a question about your taxes..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if "summary" in st.session_state:
            response = tax_qa_assistant_respond(user_input)
        else:
            response = assistant_respond_with_llm(user_input)

        cleaned_response = reformat_text_with_llm(response)
        st.markdown(cleaned_response)

    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
