import pypdf


def extract_content():
    reader = pypdf.PdfReader("Birlestirilmis_Genelgeler.pdf")

    page_count = len(reader.pages)
    extracted = ""
    for i in range(page_count):
        page = reader.pages[i]
        text = page.extract_text()
        if text:
            extracted += text + "\n"

    with open("extracted.txt", "w") as f:
        extracted = fix_genel_no_bs(extracted)
        f.write(extracted)


def fix_genel_no_bs(text) -> str:
    return text.replace("GENEL NO", "GENELGE NO")


if __name__ == "__main__":
    extract_content()
