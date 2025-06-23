from langchain.schema import Document
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import dotenv_values
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from rich.console import Console
from rich.markdown import Markdown


GEMINI_API_KEY = dotenv_values(".env")["GEMINI_API_KEY"]


with open("tnb_genelgeler_rag.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    Document(page_content=item["content"], metadata=item["metadata"]) for item in data
]

# vectorizer
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# faiss vector search and save for reuse
vector_db = FAISS.from_documents(documents, embedding_model)
vector_db.save_local("faiss_index")


# faiss vector search / bm25 keyword search
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(documents)

# ensemble
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

'''
CUSTOM_PROMPT = PromptTemplate(
    template="""
    Aşağıdaki belgelerde verilen bilgilere dayanarak soruyu yalnızca Türkçe yanıtlayın.
    Eğer belgeler soruyu yanıtlamak için yeterli değilse, sadece şu cümleyi yazın:
    "Bu konuda yeterli bilgi bulunamadı."

    Belgeler:
    {context_str}

    Soru: {question}

    Lütfen cevabı kesin ve noktalı maddeler halinde, açık ve sade Türkçe ile yazın:
    """,
    input_variables=["context_str", "question"],
)'''


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    chain_type="refine",
    verbose=True,
)
"""chain_type_kwargs={
    "question_prompt": CUSTOM_PROMPT
},  # refine -> question prompt / refine prompt"""

result = qa_chain.invoke(
    {"query": "Araç alım ve satım işlemlerinde hangi kimlikler/belgeler istenir?"}
)

console = Console()
md = Markdown(result["result"])
console.print(md)
