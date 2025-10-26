from langchain.schema import Document
import json
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import dotenv_values
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import sys


USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

if not USE_LOCAL_LLM:
    try:
        env_vars = dotenv_values(".env")
        GEMINI_API_KEY = env_vars.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        print("Please create a .env file with your GEMINI_API_KEY")
        sys.exit(1)


documents = []

try:
    with open("tnb_genelgeler_rag.json", "r", encoding="utf-8") as f:
        genelge_data = json.load(f)
    print(f"✅ Loaded {len(genelge_data)} chunks from tnb_genelgeler_rag.json")
    
    for item in genelge_data:
        # Ensure metadata has source_type for genelge
        if "source_type" not in item["metadata"]:
            item["metadata"]["source_type"] = "genelge"
        documents.append(Document(page_content=item["content"], metadata=item["metadata"]))
    
except FileNotFoundError:
    print("⚠️  tnb_genelgeler_rag.json not found. Please run process.py first.")

try:
    with open("noterlik_kanunu_rag.json", "r", encoding="utf-8") as f:
        kanun_data = json.load(f)
    print(f"✅ Loaded {len(kanun_data)} chunks from noterlik_kanunu_rag.json")
    
    for item in kanun_data:
        documents.append(Document(page_content=item["content"], metadata=item["metadata"]))
    
except FileNotFoundError:
    print("⚠️  noterlik_kanunu_rag.json not found. Please run extract_kanun.py and process_kanun.py first.")

if not documents:
    print("❌ No documents loaded. Please prepare data files first.")
    sys.exit(1)

print(f"📚 Total documents loaded: {len(documents)}")

print("🔄 Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={'batch_size': 32}
)

faiss_index_path = "faiss_index"
if os.path.exists(faiss_index_path):
    print(f"✅ Loading existing FAISS index from {faiss_index_path}...")
    vector_db = FAISS.load_local(
        faiss_index_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print(f"✅ FAISS index loaded successfully!")
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
    retrievers=[bm25_retriever, vector_retriever], 
    weights=[0.5, 0.5]
)

if USE_LOCAL_LLM:
    print("🔄 Initializing Local LLM (Ollama)...")
    llm = Ollama(
        model="llama3.2:3b",
        temperature=0.3
    )
    print("✅ Local LLM initialized (Ollama - Llama 3.2 3B)")
else:
    print("🔄 Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    print("✅ Gemini LLM initialized")

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
    template=turkish_legal_prompt,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    chain_type="stuff",  # More accurate and faster than "refine"
    chain_type_kwargs={
        "prompt": prompt_template,
        "document_separator": "\n---\n"
    },
    return_source_documents=True,  # Return sources for transparency
    verbose=True,
)

print("✅ RAG system initialized successfully!\n")


def query_rag(question: str):
    console = Console()
    
    console.print(Panel(f"[bold cyan]Soru:[/bold cyan] {question}", expand=False))
    print()
    
    try:
        result = qa_chain.invoke({"query": question})
        
        console.print("[bold green]Yanıt:[/bold green]")
        md = Markdown(result["result"])
        console.print(md)
        print()
        
        if "source_documents" in result and result["source_documents"]:
            console.print("[bold yellow]Kaynaklar:[/bold yellow]")
            for i, doc in enumerate(result["source_documents"][:3], 1):
                metadata = doc.metadata
                source_type = metadata.get('source_type', 'genelge')
                
                if source_type == 'kanun':
                    madde_no = metadata.get('madde_no', 'N/A')
                    madde_baslik = metadata.get('madde_baslik', '')
                    title = f"{i}. Noterlik Kanunu - Madde {madde_no}"
                    if madde_baslik:
                        title += f" ({madde_baslik})"
                    console.print(title)
                    kisim = metadata.get('kisim', '')
                    if kisim:
                        console.print(f"   {kisim}")
                else:
                    console.print(f"{i}. Genelge {metadata.get('genelge_no')} - Madde {metadata.get('madde_no')}")
                    console.print(f"   {metadata.get('genelge_baslik', 'N/A')}")
                
                console.print(f"   {doc.page_content[:150]}...\n")
        
        return result
    
    except Exception as e:
        console.print(f"[bold red]❌ Hata:[/bold red] {e}")
        return None


if __name__ == "__main__":
    result = query_rag("Araç satış işlemlerinde hangi belgeler gereklidir?")
