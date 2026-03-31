from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text() or " "
    return text

def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.from_texts(chunks, embeddings)
    return db

def get_conversation(vectorstore):
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    llm=HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.1-8B-Instruct',
        temperature=0.5,
        max_new_tokens=512,
    )

    prompt=PromptTemplate(template="""
    You are a helpful assistant.

    Answer the question ONLY using the provided context from the PDFs.

    If the answer is not in the context, say:
    "This information is not available in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["question","context"])

    model=ChatHuggingFace(llm=llm)
    conversation=ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt":prompt}
    )
    return conversation

def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')
    st.header('Chat with multiple PDFs')
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question=st.chat_input("Ask a question about your documents")
  
    if user_question:
        st.chat_message("user").markdown(user_question)

        st.session_state.chat_history.append({
            "role":"user",
            "content":user_question
        })

        if st.session_state.conversation:
            response=st.session_state.conversation({"question":user_question})
            answer=response["answer"]

            st.chat_message("assistant").markdown(answer)

            st.session_state.chat_history.append({
                "role":"assistant",
                "content":answer
            })
        else:
            st.warning("Please upload and process PDFs first.")

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                progress = st.progress(0)
                status = st.empty()

                with st.spinner("Processing documents..."):
                    status.text("📄 Reading PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    progress.progress(25)

                    status.text("✂️ Splitting into chunks...")
                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    ).split_text(raw_text)
                    progress.progress(50)

                    status.text("🧠 Creating embeddings...")
                    vectorStore = get_vectorstore(chunks)
                    progress.progress(75)

                    status.text("🔗 Building conversation chain...")
                    st.session_state.conversation = get_conversation(vectorStore)
                    progress.progress(100)

                status.text("✅ Done! You can now ask questions.")
                st.success("Documents processed successfully!")


if __name__=='__main__':
    main()