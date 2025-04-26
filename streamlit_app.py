import streamlit as st
import os
from dotenv import load_dotenv
import groq
from PyPDF2 import PdfReader
import glob

# Load environment variables
load_dotenv()

# Initialize Groq client
client = groq.Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def load_knowledge_base():
    knowledge_base = ""
    
    # Define the knowledge directory
    knowledge_dir = "/Users/rithikak/Desktop/Final Data"
    
    # Create directory if it doesn't exist
    if not os.path.exists(knowledge_dir):
        os.makedirs(knowledge_dir)
        return "No knowledge files found. Please add PDF or TXT files to the knowledge directory."

    # Process PDF files
    for pdf_file in glob.glob(os.path.join(knowledge_dir, "*.pdf")):
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                knowledge_base += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF file {pdf_file}: {str(e)}")

    # Process TXT files
    for txt_file in glob.glob(os.path.join(knowledge_dir, "*.txt")):
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                knowledge_base += file.read() + "\n"
        except Exception as e:
            st.error(f"Error reading text file {txt_file}: {str(e)}")

    return knowledge_base

def get_college_info(query):
    system_message = """You are a helpful assistant. Based on the provided knowledge respond accurately, and dont give any assumptions."""
    
    try:
        response_placeholder = st.empty()
        full_response = ""
        
        # Create chat completion with streaming
        for chunk in client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
            stream=True  # Enable streaming
        ):
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Vellore Colleges Chatbot", page_icon="🎓")


st.title("🎓 Vellore Colleges Chatbot")
st.markdown("""
This chatbot can help you with information about colleges in Vellore, Tamil Nadu.
Ask questions about:
- Admission processes
- Courses offered
- Campus facilities
- And more!
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
# Update the chat input section
if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_college_info(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot provides information about colleges in Vellore, including:
    - VIT University
    - CMC Vellore
    - Auxilium College
    - And more...
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

#  """You are a helpful assistant that provides information about colleges in Vellore, Tamil Nadu, India. 
#    You should provide accurate information about admission processes, courses offered, campus facilities, and other relevant details."""