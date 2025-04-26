import streamlit as st
import os
import glob
import re # Import regular expressions for answer checking
import groq # For direct API calls

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory, ChatMessageHistory # Import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage # Import message types

# --- Configuration ---
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(WORKING_DIR, "vector_db_dir") # Directory where Chroma DB is stored/created
KNOWLEDGE_BASE_DIR = "/Users/rithikak/Desktop/Final Data" # !IMPORTANT: Set your source document path here
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# Phrases indicating the RAG system couldn't find the answer in the context
RAG_FAILURE_PHRASES = [
    "do not contain",  # more generic
    "provided context does not contain",
    "not mentioned",
    "context does not mention",
    "don't have information",
    "don't know",
    "unable to find",
    "no information",
    "cannot answer",
    "not available in the context",
    "no data found"
]

RAG_FAILURE_PATTERN = re.compile('|'.join(re.escape(phrase) for phrase in RAG_FAILURE_PHRASES), re.IGNORECASE)

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = st.secrets['GROQ_API_KEY']

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize Groq client for direct calls
try:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# --- LangChain Setup ---

def load_and_split_documents(source_dir):
    # (Same as before - remains unchanged)
    """Loads documents from PDF and TXT files and splits them into chunks."""
    documents = []
    # Load PDFs
    for pdf_file in glob.glob(os.path.join(source_dir, "*.pdf")):
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error loading PDF {os.path.basename(pdf_file)}: {e}. Skipping.")

    # Load TXTs
    for txt_file in glob.glob(os.path.join(source_dir, "*.txt")):
        try:
            loader = TextLoader(txt_file, encoding='utf-8') # Specify encoding
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error loading TXT {os.path.basename(txt_file)}: {e}. Skipping.")

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

@st.cache_resource # Cache the vector store loading/creation process
def setup_vectorstore():
    # (Same as before - remains unchanged)
    """Loads the persisted Chroma vector store or creates it if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
         st.info("Loading existing vector store...")
         try:
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY,
                                 embedding_function=embeddings)
            st.success("Vector store loaded successfully.")
            return vectorstore
         except Exception as e:
             st.error(f"Error loading existing vector store: {e}")
             st.info("Attempting to recreate the vector store.")
             # Fall through
    st.warning(f"Vector store not found at: {PERSIST_DIRECTORY}. Attempting to create...")

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        st.error(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        st.stop()

    if not any(fname.endswith(('.pdf', '.txt')) for fname in os.listdir(KNOWLEDGE_BASE_DIR)):
         st.error(f"No PDF or TXT files found in {KNOWLEDGE_BASE_DIR}. Cannot create vector store.")
         st.stop()

    with st.spinner(f"Loading and splitting documents from {KNOWLEDGE_BASE_DIR}..."):
        split_docs = load_and_split_documents(KNOWLEDGE_BASE_DIR)

    if not split_docs:
        st.error("Failed to load any documents.")
        st.stop()

    st.info(f"Loaded and split {len(split_docs)} document chunks.")

    with st.spinner(f"Creating and persisting vector store using '{EMBEDDING_MODEL}'. This may take a while..."):
        try:
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            st.success("Vector store created and persisted successfully!")
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            st.stop()

# Function to create Langchain memory from Streamlit messages
def get_chat_memory():
    chat_message_history = ChatMessageHistory()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_message_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            chat_message_history.add_ai_message(msg["content"])
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_message_history,
        output_key="answer",
        return_messages=True
    )


# Use @st.cache_resource for the LLM object if desired, but chain creation depends on memory which changes
# @st.cache_resource
def get_llm():
     return ChatGroq(model=LLM_MODEL,
                    temperature=0.7,
                    groq_api_key=GROQ_API_KEY,
                    max_tokens=1024)


# Chain creation now depends on memory, so we don't cache it directly with @st.cache_resource
# We will create it when needed, using cached components (vectorstore, llm)
def create_rag_chain(_vectorstore, _llm, _memory):
    """Creates the ConversationalRetrievalChain."""
    if _vectorstore is None:
        st.error("Vector store is not available. Cannot create RAG chain.")
        return None # Return None instead of stopping
    if _llm is None:
        st.error("LLM is not available. Cannot create RAG chain.")
        return None

    try:
        retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})

        _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# In create_rag_chain function

        # Relaxed QA Prompt
        qa_template = """You are a helpful AI assistant answering questions about colleges in Vellore, Tamil Nadu, India.
        Use the following pieces of retrieved context as your primary source of information to answer the question.
        Synthesize the information found in the context to provide a comprehensive answer.
        If, after reviewing the context, you genuinely cannot find the specific information needed to answer the question, clearly state that the provided documents do not contain that detail.
        Do not add information not present in the context or make assumptions.

        Context:
        {context}

        Question:
        {question}

        Answer based on provided context:""" # Changed the label slightly
        QA_PROMPT = PromptTemplate.from_template(qa_template)

        chain = ConversationalRetrievalChain.from_llm(
            llm=_llm,
            retriever=retriever,
            memory=_memory,
            # condense_question_prompt=CONDENSE_QUESTION_PROMPT, # Using default for now
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True, # Keep this to potentially inspect sources later
            verbose=False
        )
        return chain

    except Exception as e:
        st.error(f"Error creating RAG chat chain: {e}")
        return None # Return None

# --- Direct LLM Call Function ---
def stream_direct_llm_response(user_prompt, chat_history_for_llm):
    """Calls Groq API directly and streams the response."""
    system_message = "You are a helpful assistant, knowledgeable about colleges in Vellore, Tamil Nadu, India. Don't give information about all colleges and only give for Engineering colleges or Arts/Science collegees only. Answer the user's question clearly and concisely. If suppose any out of vellore question, answer them that you dont have accees to them. Also for more generic questions like lsit the colleges, give them the required answers. Also give them the required answers for suggesting colleges for any particular course they ask for"
    messages = [{"role": "system", "content": system_message}]
    # Add history (optional but good for context)
    messages.extend(chat_history_for_llm)
    # Add current user prompt
    messages.append({"role": "user", "content": user_prompt})

    try:
        response_stream = groq_client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            temperature=0.5,
            max_tokens=1024,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except groq.APIError as e:
         yield f"\n\nError communicating with Groq API (direct call): {e}"
    except Exception as e:
         yield f"\n\nAn error occurred during direct LLM streaming: {str(e)}"


# --- Streamlit UI ---
st.set_page_config(page_title="CampusConnect", page_icon="🎓", layout="centered")

st.title("🎓 Vellore Colleges Chatbot")
st.markdown("""
This chatbot first tries to answer using a local knowledge base about Vellore colleges.
If the specific information isn't found there, it will try to answer using its general knowledge.
""")

# --- Initialize Session State & Load/Create Vector Store ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

# Initialize LLM (can be cached)
if "llm" not in st.session_state:
     st.session_state.llm = get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input and Agentic Response ---

if prompt := st.chat_input("Ask your question here..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare for assistant response
    with st.chat_message("assistant"):
        final_answer = ""
        response_container = st.empty() # Placeholder for streaming output

        # 1. Try RAG Chain
        rag_success = False
        try:
            # Create memory and chain for this specific turn
            current_memory = get_chat_memory()
            rag_chain = create_rag_chain(st.session_state.vectorstore, st.session_state.llm, current_memory)

            if rag_chain:
                with st.spinner("Searching knowledge base..."):
                    # Invoke the chain - getting the full response first to evaluate it
                    rag_result = rag_chain.invoke({"question": prompt})
                    initial_rag_answer = rag_result.get("answer", "")
                    # source_documents = rag_result.get("source_documents", []) # Can use if needed

                # 2. Evaluate RAG Answer
                if initial_rag_answer and not RAG_FAILURE_PATTERN.search(initial_rag_answer):
                    # RAG answer seems useful, stream it
                    response_container.markdown("*(Retrieving from knowledge base...)*") # Indicate source
                    final_answer = initial_rag_answer # Use the already generated answer
                    response_container.markdown(final_answer) # Display non-streamed for simplicity here, or re-stream if desired
                    rag_success = True
                    # Optional: Add source info if desired
                    # if source_documents:
                    #    with st.expander("Sources"):
                    #        for doc in source_documents:
                    #            st.caption(f"- {doc.metadata.get('source', 'Unknown')}")
                else:
                    st.info("Information not found in local documents. Asking general knowledge model...")

            else:
                 st.error("RAG chain could not be created. Falling back to direct LLM call.")

        except Exception as e:
            st.error(f"Error during RAG attempt: {e}. Falling back to direct LLM call.")
            # Ensure rag_success remains False

        # 3. Fallback to Direct LLM Call (if RAG failed or wasn't useful)
        if not rag_success:
            response_container.markdown("*(Using general knowledge...)*") # Indicate source
            # Prepare history in the format the direct API expects
            direct_api_history = []
            for msg in st.session_state.messages[:-1]: # Exclude the current user prompt
                 direct_api_history.append({"role": msg["role"], "content": msg["content"]})

            # Use st.write_stream for the fallback
            try:
                streamed_response = st.write_stream(stream_direct_llm_response(prompt, direct_api_history))
                final_answer = streamed_response
            except Exception as e:
                error_message = f"An error occurred with the direct LLM call: {str(e)}"
                st.error(error_message)
                final_answer = error_message # Store error as the final answer for history

        # 4. Add the final assistant answer to history
        if final_answer:
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        else:
             # Handle case where both attempts failed somehow
             fallback_error_msg = "Sorry, I couldn't process your request through either method."
             st.error(fallback_error_msg)
             st.session_state.messages.append({"role": "assistant", "content": fallback_error_msg})


# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) first. If specific info isn't found in the loaded documents, it uses the LLM's general knowledge.
    """)
    st.markdown("---")
    st.header("Knowledge Base")
    st.markdown(f"**Source Docs:** `{KNOWLEDGE_BASE_DIR}`")
    st.markdown(f"**Vector Store:** `{PERSIST_DIRECTORY}`")
    st.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**LLM:** `{LLM_MODEL}` (via Groq)")
    st.markdown("---")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        # No need to explicitly clear chain/memory here as they are recreated each turn now
        st.rerun()

    st.markdown("---")
    st.warning("Re-vectorizing will delete the current store and rebuild it from source documents.")
    if st.button("⚠️ Re-vectorize Documents"):
        if os.path.exists(PERSIST_DIRECTORY):
            import shutil
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                st.info(f"Deleted existing vector store at {PERSIST_DIRECTORY}.")
                st.cache_resource.clear() # Clear all cached resources
                # Clear relevant session state items that might hold old objects
                keys_to_delete = ["vectorstore", "llm", "conversational_chain"] # conversational_chain might not exist anymore
                for key in keys_to_delete:
                    if key in st.session_state: del st.session_state[key]
                st.success("Cleared cache and session state. Restarting to initiate vectorization...")
                st.rerun() # Rerun triggers setup_vectorstore check

            except Exception as e:
                st.error(f"Could not delete vector store directory: {e}")
        else:
            st.info("Vector store directory not found, no need to delete.")
            st.cache_resource.clear()
            keys_to_delete = ["vectorstore", "llm", "conversational_chain"]
            for key in keys_to_delete:
                 if key in st.session_state: del st.session_state[key]
            st.rerun()
