import streamlit as st
import PyPDF2
import io
import os
#from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load API key from .env
#load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="AI Resume Critiquer (Groq)", page_icon="📃", layout="centered")

st.title("AI Resume Critiquer (Groq LLMs - Free)")
st.markdown("Upload your resume and get AI-powered feedback using **Groq-hosted LLMs** (LLaMA 3, Mistral, Gemma).")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're targeting (optional)")

analyze = st.button("Analyze Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("File does not have any content...")
            st.stop()

        prompt = f"""Please analyze this resume and provide constructive feedback. 
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}
        
        Resume content:
        {file_content}
        
        Please provide your analysis in a clear, structured format with specific recommendations."""

        # Use Groq LLM (choose model: "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it")
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-70b-8192",  # High quality
            temperature=0.7,
            max_tokens=1000
        )

        response = llm.invoke([
            SystemMessage(content="You are an expert resume reviewer with years of experience in HR and recruitment."),
            HumanMessage(content=prompt)
        ])

        st.markdown("### Analysis Results")
        st.markdown(response.content)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")







