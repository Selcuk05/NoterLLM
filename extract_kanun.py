import pypdf


def extract_kanun():
    reader = pypdf.PdfReader("documents/Noterlik_Kanunu.pdf")

    page_count = len(reader.pages)
    extracted = ""
    
    print(f"Noterlik Kanunu okunuyor... ({page_count} sayfa)")
    
    for i in range(page_count):
        page = reader.pages[i]
        text = page.extract_text()
        if text:
            extracted += text + "\n"

    with open("kanun_extracted.txt", "w", encoding="utf-8") as f:
        f.write(extracted)
    
    print(f"Toplam karakter sayısı: {len(extracted):,}")


if __name__ == "__main__":
    extract_kanun()

