# 📄 PDFSensei

Chat with your PDFs using RAG, LangChain, and HuggingFace.

---

## 🚀 Overview

DocuChat is an AI-powered app where you can upload PDF documents and ask questions about them. It uses Retrieval-Augmented Generation (RAG) to provide answers strictly based on your documents.

---

## ✨ Features

* Upload multiple PDFs
* Ask questions in natural language
* Context-based answers using RAG
* Fast search using FAISS
* HuggingFace LLM integration
* Clean chat UI with Streamlit
* Avoids answering outside PDF content

---

## 🛠️ Tech Stack

* Frontend: Streamlit
* Backend: Python
* LLM: HuggingFace (FLAN-T5)
* Embeddings: Sentence Transformers
* Vector Store: FAISS
* Framework: LangChain

---

## ⚙️ Installation

git clone https://github.com/Nikk118/PDFSensei.git
cd PDFSensei
pip install -r requirements.txt

---

## 🔐 Environment Variables

Create a `.env` file and add:

HUGGINGFACEHUB_API_TOKEN=your_api_key_here

---

## ▶️ Run the App

streamlit run app.py

---

## 📌 How It Works

1. Upload PDFs
2. Extract and split text
3. Convert text into embeddings
4. Store in FAISS
5. Retrieve relevant chunks
6. Generate answer using LLM

---

## ⚠️ Limitations

* First run may take time (model download)
* HuggingFace API can be slow sometimes
* Answers depend on PDF content

---

## 🚀 Future Improvements

* Streaming responses
* Source highlighting
* Better UI
* Faster LLM integration

---

## 🤝 Contributing

Feel free to fork and improve this project.

---

## 📄 License

MIT License
