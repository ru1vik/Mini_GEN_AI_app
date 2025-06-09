import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import shutil
import gc

# --- MODEL SETTINGS ---
MODEL_NAME = "google/flan-t5-base"  # No API key required, chat tuned
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Small & fast

# --- LOAD LLM MODEL ---
@st.cache_resource
def model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    return pipe

# --- PROCESS PDF AND BUILD VECTORSTORE ---
def process_pdf(pdf_path, embedder, rebuild_db):
    if rebuild_db:
        try:
            if os.path.exists("db"):
                gc.collect()  # try to force release
                shutil.rmtree("db")
                print("DB cleared.")
        except Exception as e:
            print(f"Could not delete DB folder: {e}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
    texts,
    embedder

    )
    vectorstore.persist()

    return vectorstore

# --- BUILD PROMPT ---
def build_prompt(context, user_question):
    prompt_template = prompt_template = """
Answer the following question based on the provided context.

Context:
{context}


"""
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt.format(context=context, question=user_question)

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="📚 Chat with your PDF (RAG App)", page_icon="📚")
    st.title("📚 Chat with your PDF")

    st.sidebar.header("Upload & Settings")

    # Upload PDF
    uploaded_pdf = st.sidebar.file_uploader("Upload your PDF", type="pdf")

    # Reset DB button
    rebuild_db = st.sidebar.checkbox("Reset vector DB (force re-embed)", value=False)

    if uploaded_pdf is not None:
        pdf_path = os.path.join("uploaded_pdf", uploaded_pdf.name)
        os.makedirs("uploaded_pdf", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.sidebar.success("PDF uploaded!")
    if not os.path.exists("db"):
        st.sidebar.warning("No existing vector DB found. Rebuilding...")
        rebuild_db = True

        # Load embedding model
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Process PDF
        vectorstore = process_pdf(pdf_path, embedder, rebuild_db)

        # Load LLM pipeline
        llama_pipe = model()

        # Conversation history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat window
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])

        # User input
        user_input = st.chat_input("Ask a question about your PDF:")
        if user_input:
            # Display user message
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Retrieve top 3 relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build prompt
            full_prompt = build_prompt(context, user_input)

            # Generate response
            response = llama_pipe(full_prompt)[0]["generated_text"].strip()

            # Display assistant message
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
