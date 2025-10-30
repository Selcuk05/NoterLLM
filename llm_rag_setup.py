from langchain.schema import Document
import json
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Optional

_qa_chain: Optional[RetrievalQA] = None
_initialized = False


def init_rag():
    global _qa_chain, _initialized

    if _initialized:
        return

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print(
            "⚠️  HF_TOKEN not found in environment variables. Set it in Spaces secrets or .env file"
        )

    documents = []

    try:
        with open("tnb_genelgeler_rag.json", "r", encoding="utf-8") as f:
            genelge_data = json.load(f)
        print(f"✅ Loaded {len(genelge_data)} chunks from tnb_genelgeler_rag.json")

        for item in genelge_data:
            if "source_type" not in item.get("metadata", {}):
                item.setdefault("metadata", {})["source_type"] = "genelge"
            documents.append(
                Document(
                    page_content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                )
            )

    except FileNotFoundError:
        print("⚠️  tnb_genelgeler_rag.json not found. Please upload data files.")

    try:
        with open("noterlik_kanunu_rag.json", "r", encoding="utf-8") as f:
            kanun_data = json.load(f)
        print(f"✅ Loaded {len(kanun_data)} chunks from noterlik_kanunu_rag.json")

        for item in kanun_data:
            documents.append(
                Document(
                    page_content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                )
            )

    except FileNotFoundError:
        print("⚠️  noterlik_kanunu_rag.json not found. Please upload data files.")

    if not documents:
        print("❌ No documents loaded. Please prepare data files first.")
        _initialized = False
        return

    print(f"📚 Total documents loaded: {len(documents)}")

    faiss_index_path = "faiss_index"

    print("🔄 Initializing embedding model (multilingual-e5-base)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base", encode_kwargs={"batch_size": 32}
    )
    print("✅ Embedding model initialized")

    # Load or create FAISS index
    if os.path.exists(faiss_index_path):
        print(f"✅ Found existing FAISS index at {faiss_index_path} — loading...")
        try:
            vector_db = FAISS.load_local(
                faiss_index_path, embedding_model, allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            print(f"🔄 Creating new FAISS index...")
            vector_db = FAISS.from_documents(documents, embedding_model)
            vector_db.save_local(faiss_index_path)
            print(f"✅ FAISS index created and saved to {faiss_index_path}")
    else:
        print(f"🔄 Creating new FAISS index (this may take a few minutes)...")
        vector_db = FAISS.from_documents(documents, embedding_model)
        vector_db.save_local(faiss_index_path)
        print(f"✅ FAISS index created and saved to {faiss_index_path}")

    bm25_path = "bm25_retriever.pkl"
    if os.path.exists(bm25_path):
        print(f"✅ Loading existing BM25 index from {bm25_path}...")
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        print(f"✅ BM25 index loaded successfully!")
    else:
        print(f"🔄 Creating new BM25 index...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"✅ BM25 index created and saved to {bm25_path}")

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    print("🔄 Initializing HuggingFace LLM (Qwen2.5-7B-Instruct)...")
    try:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=1024,
            top_p=0.95,
            repetition_penalty=1.1,
        )

        llm = ChatHuggingFace(llm=llm_endpoint)
        print("✅ HuggingFace LLM initialized (Qwen2.5-7B-Instruct)")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        print(f"   HF_TOKEN is {'set' if HF_TOKEN else 'NOT set'}")
        _initialized = False
        return

    turkish_legal_prompt = """Sen Türk Noter Hukuku konusunda uzman bir yapay zeka asistanısın. Görevin, Noterlik Kanunu ve Türkiye Noterler Birliği genelgelerinden yararlanarak kullanıcının sorusunu doğru ve eksiksiz yanıtlamaktır.

    BAĞLAM BİLGİLERİ (Kanun ve Genelgelerden):
    {context}

    KULLANICI SORUSU: {question}

    YANITLAMA STRATEJİSİ:
    1. **KAYNAK ÖNCELİĞİ**: 
       - Noterlik Kanunu → Temel yasal çerçeve ve genel kurallar
       - TNB Genelgeleri → Kanunun uygulanmasına ilişkin özel düzenlemeler ve açıklamalar
       - Her iki kaynağı da kontrol et ve ilgili olanları kullan

    2. **HİBRİT YANITLAMA**: 
       - Kanun maddeleri varsa bunları temel al
       - Genelgelerdeki uygulama detayları varsa ekle
       - Kaynak belirtmeyi unutma!

    3. **KAYNAK BELİRTME**:
       - Kanundan alınan bilgi → "Noterlik Kanunu Madde X'e göre..."
       - Genelgelerden alınan bilgi → "Genelge X, Madde Y'ye göre..."
       - Genel bilgi → "Genel olarak..." veya "Türk Hukuku'nda..."

    4. **KALİTE KURALLARI**:
       - Yanıtını net, anlaşılır ve yapılandırılmış şekilde sun
       - Hukuki terminolojiyi doğru kullan
       - Kesin olmadığın konularda varsayımda bulunma
       - Hem kanunu hem genelgeleri kaynak olarak kullanabilirsin

    YANITINIZ:"""

    prompt_template = PromptTemplate(
        template=turkish_legal_prompt, input_variables=["context", "question"]
    )

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template, "document_separator": "\n---\n"},
        return_source_documents=True,
        verbose=False,
    )

    if _qa_chain is None:
        print("❌ QA Chain creation failed silently")
        return

    print("✅ RAG system initialized successfully!\n")
    _initialized = True


def query_rag(question: str):
    global _qa_chain, _initialized

    if not _initialized:
        init_rag()

    if not _initialized or _qa_chain is None:
        print("❌ RAG system is not properly initialized. Chain or data missing.")
        return None

    try:
        print(f"DEBUG: _qa_chain type: {type(_qa_chain)}")
        print(f"DEBUG: _qa_chain.invoke type: {type(_qa_chain.invoke)}")
        print(f"DEBUG: Calling invoke with question: {question[:50]}...")
        result = _qa_chain.invoke({"query": question})
        return result
    except Exception as e:
        print(f"❌ Error querying RAG: {e}")
        import traceback

        traceback.print_exc()
        return None
