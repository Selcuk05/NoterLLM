import json
from dotenv import dotenv_values
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.chains import RetrievalQA
import streamlit as st


def rag_setup():
    GEMINI_API_KEY = dotenv_values(".env")["GEMINI_API_KEY"]

    with open("tnb_genelgeler_rag.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in data
    ]

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_db = FAISS.load_local(
        "faiss_index", embedding_model, allow_dangerous_deserialization=True
    )

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(documents)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble_retriever,
        chain_type="refine",
        verbose=True,
    )

    return qa_chain


if __name__ == "__main__":
    qa_chain = rag_setup()

    st.title("NoterLLM")

    prompt = st.chat_input("Say something")
    if prompt:
        try:
            result = qa_chain.invoke({"query": prompt})
        except:
            st.toast("Bir hata oluştu. Lütfen daha sonra tekrar deneyin.", icon="🚨")
        answer = result["result"]
        st.markdown(answer)
