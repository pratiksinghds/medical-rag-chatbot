import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def load_llm():
        return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.5,
        max_output_tokens=512,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

#2 Connect LLM with FAISS and create chain
DB_FAISS_PATH="vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE="""
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])