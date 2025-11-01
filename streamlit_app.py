
import streamlit as st  
from functions import *
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize API key from environment or Streamlit secrets
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def display_pdf(uploaded_file):

    """
    Display a PDF file that has been uploaded to Streamlit.

    Uses Streamlit's built-in st.pdf() function which works reliably
    both locally and on Streamlit Cloud.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded PDF file to display.

    Returns
    -------
    None
    """
    # Use Streamlit's built-in PDF viewer - works on both local and hosted environments
    st.pdf(uploaded_file, height=1000)


def load_streamlit_page():

    """
    Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_file: The uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        # st.header("Input your OpenAI API key")
        # st.text_input('OpenAI API key', type='password', key='api_key',
        #             label_visibility="collapsed", disabled=False)
        st.header("RAG PDF System")
        st.markdown("Upload a PDF document and extract structured data using OpenAI's LLM with intelligent document retrieval and verifiable answers.")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

    return col1, col2, uploaded_file


# Make a streamlit page
col1, col2, uploaded_file = load_streamlit_page()

# Process the input
if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)
    
    with col1:
        # Load in the documents
        documents = get_pdf_text(uploaded_file)
        st.session_state.vector_store = create_vectorstore_from_texts(documents, 
                                                                      api_key=st.session_state.api_key,
                                                                      file_name=uploaded_file.name)
        st.success("Input Processed")

# Extract structured data
with col1:
    if st.button("Extract Paper Information"):
        with st.spinner("Extracting paper metadata..."):
            answer = query_document(vectorstore = st.session_state.vector_store, 
                                    query = "Give me the title, summary, publication date, and authors of the research paper.",
                                    api_key = st.session_state.api_key)
                            
            st.write(answer)
