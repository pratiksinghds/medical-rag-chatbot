# 🩺 Clinical Q&A Assistant (Domain-Specific Medical RAG)

An end-to-end Retrieval-Augmented Generation (RAG) assistant designed to parse unstructured clinical texts, extract relevant medical passages, and generate grounded answers with verifiable source attributions.

---

## 🌟 Key Features
* **Semantic Document Retrieval:** Chunks large medical texts and builds high-density vector representations using Hugging Face sentence transformers.
* **Grounded Generation:** Enforces strict prompt boundaries to mitigate medical hallucination, instructing the model to rely only on retrieved contexts.
* **Traceable Citations:** Returns page numbers and document snippets in an interactive Streamlit UI for clinical auditability.

---

## 🏗️ Architecture & Tech Stack
* **LLM Engine:** Google Gemini API (`gemini-1.5-flash`) via `langchain-google-genai`
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Index:** FAISS (Facebook AI Similarity Search)
* **Frameworks:** LangChain (`RetrievalQA`, custom prompt engineering)
* **Frontend:** Streamlit

---

## 🚀 Setup & Execution

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/pratiksinghds/medical-rag-chatbot.git](https://github.com/pratiksinghds/medical-rag-chatbot.git)
cd medical-rag-chatbot
pipenv install
pipenv shell
