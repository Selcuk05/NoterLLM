import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class GenelgeChunk:
    genelge_no: int
    baslik: str
    madde_no: str
    alt_madde: str
    icerik: str
    full_path: str  # Örn: "Genelge 1 > 2- Özel kanunlar > a) Ülkemizde bulunan..."
    chunk_id: str


class TNBGenelgeProcessor:
    def __init__(self):
        self.genelgeler = []
        self.chunks = []

    def parse_genelge_text(self, text: str) -> List[Dict]:
        genelge_pattern = r"GENELGE NO (\d+)\s*\n([^\n]+)"
        genelge_matches = list(re.finditer(genelge_pattern, text))

        genelgeler = []

        for i, match in enumerate(genelge_matches):
            genelge_no = int(match.group(1))
            genelge_baslik = match.group(2).strip()

            # Genelge içeriğinin başlangıç ve bitiş pozisyonları
            start_pos = match.end()
            end_pos = (
                genelge_matches[i + 1].start()
                if i + 1 < len(genelge_matches)
                else len(text)
            )

            genelge_icerik = text[start_pos:end_pos].strip()

            genelgeler.append(
                {"no": genelge_no, "baslik": genelge_baslik, "icerik": genelge_icerik}
            )

        return genelgeler

    def parse_genelge_maddeleri(self, genelge_icerik: str) -> List[Dict]:
        lines = genelge_icerik.split("\n")
        maddeler = []
        current_madde = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ana madde kontrolü
            ana_madde_match = re.match(r"^(\d+)-?\s*(.+)", line)
            if ana_madde_match:
                if current_madde:
                    maddeler.append(
                        {
                            "madde_no": current_madde,
                            "icerik": "\n".join(current_content).strip(),
                        }
                    )

                current_madde = ana_madde_match.group(1)
                current_content = [ana_madde_match.group(2)]
                continue

            alt_madde_match = re.match(r"^([a-z]+\)|[çğıöşüÇĞIİÖŞÜ]+\))\s*(.+)", line)
            if alt_madde_match and current_madde:
                current_content.append(line)
                continue

            if current_madde:
                current_content.append(line)

        if current_madde:
            maddeler.append(
                {
                    "madde_no": current_madde,
                    "icerik": "\n".join(current_content).strip(),
                }
            )

        return maddeler

    def create_chunks(self, genelge: Dict) -> List[GenelgeChunk]:
        """
        Create chunks with hierarchical context preservation.
        Each chunk includes genelge title and madde context for better retrieval.
        """
        chunks = []
        maddeler = self.parse_genelge_maddeleri(genelge["icerik"])

        for madde in maddeler:
            # Her maddeyi alt parçalara böl
            alt_chunks = self.split_madde_content(madde["icerik"])

            for i, chunk_content in enumerate(alt_chunks):
                # HIERARCHICAL CONTEXT: Prepend genelge and madde info
                # This dramatically improves retrieval accuracy by preserving context
                hierarchical_content = self._create_hierarchical_content(
                    genelge_no=genelge['no'],
                    genelge_baslik=genelge['baslik'],
                    madde_no=madde['madde_no'],
                    chunk_content=chunk_content
                )
                
                chunk_id = hashlib.md5(
                    f"{genelge['no']}-{madde['madde_no']}-{i}-{chunk_content[:50]}".encode()
                ).hexdigest()[:12]

                full_path = f"Genelge {genelge['no']} > {madde['madde_no']}- {chunk_content[:50]}..."

                chunk = GenelgeChunk(
                    genelge_no=genelge["no"],
                    baslik=genelge["baslik"],
                    madde_no=madde["madde_no"],
                    alt_madde=f"Bölüm {i+1}" if len(alt_chunks) > 1 else "",
                    icerik=hierarchical_content,  # Use hierarchical content instead of raw content
                    full_path=full_path,
                    chunk_id=chunk_id,
                )
                chunks.append(chunk)

        return chunks
    
    def _create_hierarchical_content(
        self, 
        genelge_no: int, 
        genelge_baslik: str, 
        madde_no: str, 
        chunk_content: str
    ) -> str:
        """
        Create hierarchical content that includes context for better retrieval.
        Format: [Genelge Info] > [Madde Info] > [Content]
        """
        # Build hierarchical structure
        hierarchical = f"GENELGE NO {genelge_no}: {genelge_baslik}\n"
        hierarchical += f"Madde {madde_no}\n"
        hierarchical += f"---\n"
        hierarchical += chunk_content
        
        return hierarchical

    def split_madde_content(
        self, 
        content: str, 
        max_length: int = 1500,  # Increased from 500 to 1500
        overlap: int = 200  # Added overlap for context preservation
    ) -> List[str]:
        """
        Split content into chunks with overlap for better context preservation.
        Optimized for legal documents with improved chunking strategy.
        """
        if len(content) <= max_length:
            return [content]

        # Try to split on alt madde boundaries first
        alt_madde_pattern = r"([a-z]+\)|[çğıöşüÇĞIİÖŞÜ]+\))"
        parts = re.split(alt_madde_pattern, content)

        chunks = []
        current_chunk = ""

        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                part = parts[i] + parts[i + 1]
            else:
                part = parts[i]

            # Check if adding this part would exceed max_length
            if len(current_chunk + part) <= max_length:
                current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from the end of current chunk
                    if len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + part
                    else:
                        current_chunk = part
                else:
                    # Part itself is too long, split by sentences
                    sentences = re.split(r'([.!?]\s+)', part)
                    temp_chunk = ""
                    for j in range(0, len(sentences), 2):
                        if j + 1 < len(sentences):
                            sentence = sentences[j] + sentences[j + 1]
                        else:
                            sentence = sentences[j]
                        
                        if len(temp_chunk + sentence) <= max_length:
                            temp_chunk += sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sentence
                            else:
                                # Even single sentence is too long, force split
                                chunks.append(sentence[:max_length].strip())
                                temp_chunk = sentence[max_length:]
                    
                    if temp_chunk:
                        current_chunk = temp_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def process_file(self, file_path: str) -> List[GenelgeChunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.genelgeler = self.parse_genelge_text(text)

        for genelge in self.genelgeler:
            chunks = self.create_chunks(genelge)
            self.chunks.extend(chunks)

        return self.chunks

    def export_for_rag(self, output_path: str):
        rag_data = []

        for chunk in self.chunks:
            rag_data.append(
                {
                    "id": chunk.chunk_id,
                    "content": chunk.icerik,
                    "metadata": {
                        "genelge_no": chunk.genelge_no,
                        "genelge_baslik": chunk.baslik,
                        "madde_no": chunk.madde_no,
                        "alt_madde": chunk.alt_madde,
                        "full_path": chunk.full_path,
                        "source": f"TNB Genelge {chunk.genelge_no}",
                    },
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)

        print(
            f"✅ {len(rag_data)} chunk RAG formatında {output_path} dosyasına kaydedildi"
        )

    def get_statistics(self):
        return {
            "toplam_genelge": len(self.genelgeler),
            "toplam_chunk": len(self.chunks),
            "ortalama_chunk_uzunlugu": (
                sum(len(chunk.icerik) for chunk in self.chunks) / len(self.chunks)
                if self.chunks
                else 0
            ),
            "genelge_dagilimi": {
                g["no"]: len([c for c in self.chunks if c.genelge_no == g["no"]])
                for g in self.genelgeler
            },
        }


if __name__ == "__main__":
    processor = TNBGenelgeProcessor()

    chunks = processor.process_file("extracted.txt")
    processor.export_for_rag("tnb_genelgeler_rag.json")

    stats = processor.get_statistics()
    print("\n📊 İşlem İstatistikleri:")
    print(f"Toplam Genelge: {stats['toplam_genelge']}")
    print(f"Toplam Chunk: {stats['toplam_chunk']}")
    print(f"Ortalama Chunk Uzunluğu: {stats['ortalama_chunk_uzunlugu']:.0f} karakter")
    print("\nGenelge Dağılımı:")
    for genelge_no, chunk_count in stats["genelge_dagilimi"].items():
        print(f"  Genelge {genelge_no}: {chunk_count} chunk")

    print("\n📝 Örnek Chunklar:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n{i+1}. {chunk.full_path}")
        print(f"   ID: {chunk.chunk_id}")
        print(f"   İçerik: {chunk.icerik[:100]}...")
