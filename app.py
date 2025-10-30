import gradio as gr
from llm_rag_setup import query_rag, init_rag

print("🚀 Initializing RAG system at startup...")
init_rag()
print("✅ RAG system ready!")

custom_css = """
.container {
    max-width: 1200px;
    margin: auto;
}
.source-box {
    background-color: #eef2f5;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #2196F3;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.source-box * {
    color: #2c3e50 !important;
}
.source-title {
    font-weight: 600;
    margin-bottom: 10px;
    font-size: 1.1em;
    color: #1a73e8 !important;
}
.source-content {
    font-size: 0.95em;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.sources-header {
    color: #1a73e8;
    font-size: 1.2em;
    font-weight: 600;
    margin: 20px 0 15px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #1a73e8;
}
.footer {
    text-align: center;
    margin-top: 20px;
    color: #666;
}
"""


def format_sources(source_documents):
    if not source_documents:
        return "Kaynak bulunamadı."

    sources_html = ""
    for i, doc in enumerate(source_documents[:3], 1):
        metadata = doc.metadata
        source_type = metadata.get("source_type", "genelge")

        if source_type == "kanun":
            madde_no = metadata.get("madde_no", "N/A")
            madde_baslik = metadata.get("madde_baslik", "")
            title = f"📜 Noterlik Kanunu - Madde {madde_no}"
            if madde_baslik:
                title += f" ({madde_baslik})"
            kisim = metadata.get("kisim", "")
            content = f"{kisim}\n\n{doc.page_content[:200]}..."
        else:
            genelge_no = metadata.get("genelge_no", "N/A")
            madde_no = metadata.get("madde_no", "N/A")
            title = f"📋 Genelge {genelge_no} - Madde {madde_no}"
            genelge_baslik = metadata.get("genelge_baslik", "N/A")
            content = doc.page_content[:500] + (
                "..." if len(doc.page_content) > 500 else ""
            )

        sources_html += f"""
<div class="source-box">
    <div class="source-title">{i}. {title}</div>
    <div class="source-content">{content}</div>
</div>
"""

    return sources_html


def chat_with_rag(message, history):
    if not message.strip():
        return "", history

    try:
        result = query_rag(message)
        if result is None:
            answer = "❌ Sistem başlatılamadı veya veri eksik. Lütfen sunucu günlüklerini kontrol edin."
        else:
            answer = result.get("result", "(Cevap alınamadı)")

        sources_html = ""
        if result and "source_documents" in result and result["source_documents"]:
            sources_html = (
                '<br><br><div class="sources-header">📚 Kaynaklar</div>'
                + format_sources(result["source_documents"])
            )

        full_response = answer + sources_html
        history.append((message, full_response))

        return "", history

    except Exception as e:
        error_message = f"❌ Hata oluştu: {str(e)}"
        history.append((message, error_message))
        return "", history


def clear_chat():
    return [], []


examples = [
    "Araç satış işlemlerinde hangi belgeler gereklidir?",
    "Noterlik işlemlerinde harç ve karar pulu nasıl hesaplanır?",
    "Vekaletname düzenlenirken dikkat edilmesi gereken hususlar nelerdir?",
    "Gayrimenkul satış vaadi sözleşmesi nedir?",
]


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ⚖️ NoterLLM - Türk Noter Hukuku Asistanı
        
        Noterlik Kanunu ve Türkiye Noterler Birliği genelgelerine dayalı AI destekli soru-cevap sistemi.
        
        **Kaynaklar:**
        - 📜 Noterlik Kanunu (1512) - 213 madde
        - 📋 TNB Genelgeleri - 125+ genelge
        
        *Bu sistem genelgelere dayalı bilgi sağlar ancak resmi hukuki danışmanlık yerine geçmez.*
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Sohbet Geçmişi",
                height=500,
                show_label=True,
                avatar_images=(None, None),
                bubble_full_width=False,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Sorunuz",
                    placeholder="Noterlik hukuku ile ilgili sorunuzu yazın...",
                    show_label=False,
                    scale=9,
                    container=False,
                )
                submit_btn = gr.Button("Gönder", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Sohbeti Temizle", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Örnek Sorular")
            gr.Examples(
                examples=examples,
                inputs=msg,
                label="Aşağıdaki sorulardan birini seçebilirsiniz:",
            )

            gr.Markdown(
                """
                ### ℹ️ Bilgi
                
                **Model:** Qwen2.5-7B-Instruct  
                **Embedding:** multilingual-e5-base  
                **Retrieval:** FAISS + BM25 (Hybrid)
                
                **Özellikler:**
                - 🔍 Semantic & Keyword Search
                - 📚 Kaynak Referansları
                - 🇹🇷 Türkçe Optimizasyonu
                """
            )

    submit_btn.click(
        fn=chat_with_rag,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )

    msg.submit(
        fn=chat_with_rag,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[msg, chatbot],
    )

    gr.Markdown(
        """
        <div class="footer">
            <p>Powered by HuggingFace 🤗 | Built with Gradio</p>
        </div>
        """
    )

demo.launch()
