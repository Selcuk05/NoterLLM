import streamlit as st
from llm_rag_setup import qa_chain
import sys

st.set_page_config(
    page_title="NoterLLM - Noter Hukuku Asistanı",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ NoterLLM - Türk Noter Hukuku Asistanı")
st.markdown("Türkiye Noterler Birliği genelgelerine dayalı AI destekli soru-cevap sistemi")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(f"**{source['title']}**")
                    st.caption(source['content'])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                st.markdown(answer)
                
                sources = []
                if "source_documents" in result and result["source_documents"]:
                    with st.expander("📚 Kaynaklar"):
                        for i, doc in enumerate(result["source_documents"][:3], 1):
                            metadata = doc.metadata
                            title = f"{i}. Genelge {metadata.get('genelge_no')} - Madde {metadata.get('madde_no')}"
                            content = f"{metadata.get('genelge_baslik', 'N/A')}\n\n{doc.page_content[:200]}..."
                            st.markdown(f"**{title}**")
                            st.caption(content)
                            sources.append({"title": title, "content": content})
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Hata: {e}")

with st.sidebar:
    st.header("📋 Bilgi")
    st.info("""
    **NoterLLM** Türk Noter Hukuku hakkında sorularınızı yanıtlar.
    
    Sistem genelgeleri kullanarak doğru ve kaynak referanslı yanıtlar sunar.
    """)
    
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption(f"💬 Toplam Mesaj: {len(st.session_state.messages)}")

