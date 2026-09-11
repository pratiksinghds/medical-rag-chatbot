import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context", "question"])
    return prompt


def load_llm():
        return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.5,
        max_output_tokens=512,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
         st.chat_message(message['role']).markdown(message['content'])    

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE="""
                Use the pieces of information provided in the context to answer user's question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        #setup Gemini LLM
        llm=load_llm()


        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                 st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
     
            response = qa_chain.invoke({"query": prompt})

            result = response["result"]
            source_docs = response.get("source_documents", [])

            # Display answer
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

            # Display sources neatly inside an expandable dropdown
            if source_docs:
                with st.expander("📚 View Document Sources"):
                     for i, doc in enumerate(source_docs, 1):
                        page_num = doc.metadata.get("page_label", doc.metadata.get("page", "N/A"))
                        st.markdown(f"**Source {i} (Page {page_num}):**")
                        st.caption(doc.page_content.strip()[:300] + "...")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()     