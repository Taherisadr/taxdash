import streamlit as st
import pdfplumber
import json
import requests
import re
import html
from io import BytesIO

# --- Streamlit page setup ---
st.set_page_config(page_title="AI Tax Agent", layout="centered")
st.title("📄 AI Tax Agent - W-2 Tax Return Prototype")

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
        {"role": "assistant", "content": "Hi! I'm your AI Tax Assistant 🤖. I'll help you upload your W-2, enter details, and compute your taxes. Let's get started!"}
    ]

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

    with st.spinner("🧠 Extracting fields using Mistral via OpenRouter..."):
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

# --- Basic Input Sanitization for safety ---
def sanitize_string(s):
    if not isinstance(s, str):
        return ""
    return html.escape(s.strip())

# === Tax Calculation Logic ===
def compute_tax_summary(data: dict, filing_status: str, additional_deductions: float) -> dict:
    def safe_float(value):
        if isinstance(value, str):
            # Remove commas and any other non-numeric characters (except the decimal point)
            cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.')
            try:
                return float(cleaned_value)
            except (ValueError, TypeError):
                return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    standard_deductions = {
        "single": 13850,
        "married_filing_jointly": 27700,
        "married_filing_separately": 13850,
        "head_of_household": 20800
    }

    std_deduction = standard_deductions.get(filing_status.lower(), 13850)
    std_deduction += additional_deductions

    required_fields = ["Wages (Box 1)", "Federal Income Tax Withheld (Box 2)"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        st.warning(f"⚠️ Missing expected fields from W-2: {', '.join(missing)}")

    try:
        income = safe_float(data.get("Wages (Box 1)"))
        withheld = safe_float(data.get("Federal Income Tax Withheld (Box 2)"))
        taxable_income = max(0, income - std_deduction)

        # Simplified tax brackets for 2023 (Single filer)
        # This should be expanded for other filing statuses for a real app
        if taxable_income <= 11000:
            tax = taxable_income * 0.10
        elif taxable_income <= 44725:
            tax = 1100 + (taxable_income - 11000) * 0.12
        elif taxable_income <= 95375:
            tax = 5147 + (taxable_income - 44725) * 0.22
        else:
            tax = 16290 + (taxable_income - 95375) * 0.24

        tax = max(0, tax)
        refund_or_due = round(withheld - tax, 2)
        status_msg = "You will receive a refund." if refund_or_due > 0 else "You owe additional taxes."

        return {
            "Employee Name": sanitize_string(data.get("Employee Name", "")),
            "Employer Name": sanitize_string(data.get("Employer Name", "")),
            "Filing Year": sanitize_string(str(data.get("Filing Year", ""))),
            "Filing Status": filing_status.replace("_", " ").title(),
            "Total Income": round(income, 2),
            "Standard Deduction + Additional": round(std_deduction, 2),
            "Taxable Income": round(taxable_income, 2),
            "Estimated Tax Owed": round(tax, 2),
            "Tax Withheld": round(withheld, 2),
            "Refund or Amount Due": refund_or_due,
            "Status Message": status_msg
        }

    except Exception as e:
        st.error(f"Error in tax computation: {e}")
        return {}

# === Generate HTML Tax Return ===
def generate_tax_return_html(summary: dict) -> str:
    esc = lambda x: html.escape(str(x))
    html_content = f"""
    <html>
    <head><title>Simple Tax Return - {esc(summary.get('Filing Year',''))}</title></head>
    <body>
        <h2>Simple Tax Return Summary</h2>
        <p><strong>Employee Name:</strong> {esc(summary.get('Employee Name',''))}</p>
        <p><strong>Employer Name:</strong> {esc(summary.get('Employer Name',''))}</p>
        <p><strong>Filing Year:</strong> {esc(summary.get('Filing Year',''))}</p>
        <p><strong>Filing Status:</strong> {esc(summary.get('Filing Status',''))}</p>
        <p><strong>Total Income:</strong> ${esc(summary.get('Total Income',0))}</p>
        <p><strong>Standard Deduction + Additional:</strong> ${esc(summary.get('Standard Deduction + Additional',0))}</p>
        <p><strong>Taxable Income:</strong> ${esc(summary.get('Taxable Income',0))}</p>
        <p><strong>Estimated Tax Owed:</strong> ${esc(summary.get('Estimated Tax Owed',0))}</p>
        <p><strong>Tax Withheld:</strong> ${esc(summary.get('Tax Withheld',0))}</p>
        <p><strong>Refund or Amount Due:</strong> ${esc(summary.get('Refund or Amount Due',0))}</p>
        <h3>{esc(summary.get('Status Message',''))}</h3>
    </body>
    </html>
    """
    return html_content

# === Chatbot Response ===
def assistant_respond_with_llm(user_input):
    chat_history = st.session_state.chat_history
    system_prompt = (
        "You are a friendly and knowledgeable AI Tax Assistant helping a user step-by-step with their tax return. "
        "Guide them through W-2 upload, review, deductions, calculation, and download."
    )

    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    payload = {"model": MODEL, "messages": messages}

    try:
        with st.spinner("💬 Thinking..."):
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ Failed to reach the assistant: {str(e)}"

# === QA Chatbot (Tax Info) ===
def tax_qa_assistant_respond(question: str) -> str:
    extracted = st.session_state.get("extracted_data", {})
    summary = st.session_state.get("summary", {})

    context = {
        "Extracted W-2 Data": extracted,
        "Tax Summary": summary
    }

    system_prompt = (
        "You are an AI tax expert helping the user understand their tax situation. "
        "You have access to their W-2 data and tax summary. Answer clearly and accurately based on it."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User tax data:\n{json.dumps(context, indent=2)}"},
            {"role": "user", "content": f"User question: {question}"}
        ]
    }

    try:
        with st.spinner("🧮 Thinking..."):
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ Failed to answer: {str(e)}"

# === Main App ===
st.markdown("### Step 1: Upload your W-2 PDF file")
uploaded_file = st.file_uploader("Choose a W-2 PDF to upload", type=["pdf"], key="w2_uploader")

# View 2: Show the final summary if it's already calculated
if "summary" in st.session_state:
    summary = st.session_state["summary"]
    st.subheader("📊 Tax Summary")
    st.json(summary)

    tax_return_html = generate_tax_return_html(summary)
    st.markdown("---")
    st.markdown("### 🧾 Your Simplified Tax Return Form")
    st.components.v1.html(tax_return_html, height=450, scrolling=True)

    buf = BytesIO()
    buf.write(tax_return_html.encode('utf-8'))
    buf.seek(0)
    st.download_button(
        "📥 Download Tax Return Form (HTML)", 
        data=buf,
        file_name=f"tax_return_{summary.get('Filing Year','')}.html", 
        mime="text/html"
    )
    
    if "raw_text" in st.session_state:
        with st.expander("🔍 View Extracted W-2 Text"):
            st.text(st.session_state["raw_text"])

# View 1: Show the calculation form if a file is uploaded but not yet summarized
elif uploaded_file:
    with st.spinner("📄 Reading and extracting text from W-2..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        st.session_state["raw_text"] = raw_text

    extracted_data = extract_fields_from_text(raw_text)
    if extracted_data:
        st.session_state["extracted_data"] = extracted_data
        st.markdown("### Step 2: Provide additional tax info")

        filing_status = st.selectbox("Select your Filing Status", options=[
            "single", "married_filing_jointly", "married_filing_separately", "head_of_household"])
        
        additional_deductions = st.number_input("Additional Deductions", min_value=0.0, step=100.0, value=0.0)

        if st.button("Calculate Tax"):
            with st.spinner("💰 Calculating your tax summary..."):
                summary = compute_tax_summary(extracted_data, filing_status, additional_deductions)
            if summary:
                st.session_state["summary"] = summary
                st.rerun()  # Rerun to switch to the summary view
    else:
        st.error("⚠️ Failed to extract fields from the W-2 form. Please check your file or try another.")
else:
    st.info("Please upload a W-2 PDF file to get started.")

# === Chat Interface (always at the bottom) ===
st.markdown("---")
st.subheader("🤖 Chat with your AI Tax Assistant")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
if user_input := st.chat_input("Ask a question about your taxes..."):
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and display assistant response
    with st.chat_message("assistant"):
    # Use the appropriate assistant based on context
        if "summary" in st.session_state:
            response = tax_qa_assistant_respond(user_input)
        else:
            response = assistant_respond_with_llm(user_input)
        
        formatted_response = response.replace("\n", "  \n")  # Add Markdown-compatible line breaks
        st.markdown(formatted_response)

    
    # Add assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
