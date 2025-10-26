import re
import json
from typing import List, Dict
from dataclasses import dataclass
import hashlib


@dataclass
class KanunChunk:
    madde_no: str
    madde_baslik: str
    kisim: str
    bolum: str
    icerik: str
    full_path: str
    chunk_id: str


class NoterlikKanunuProcessor:
    def __init__(self):
        self.maddeler = []
        self.chunks = []
        self.current_kisim = ""
        self.current_bolum = ""

    def parse_kanun_text(self, text: str) -> List[Dict]:
        lines = text.split('\n')
        maddeler = []
        current_madde = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # KISIM başlığı kontrolü
            kisim_match = re.match(r'^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s+KISIM\s*$', line, re.IGNORECASE)
            if kisim_match:
                self.current_kisim = line
                self.current_bolum = ""
                continue
            
            # alt başlık
            if self.current_kisim and not line.startswith('Madde') and not line.startswith('BÖLÜM'):
                # Eğer önceki satırda KISIM vardı ve bu satır Madde ile başlamıyorsa, KISIM başlığının devamı
                if not re.match(r'^[a-z]', line):  # Küçük harfle başlamıyorsa başlık olabilir
                    self.current_kisim = f"{self.current_kisim} - {line}"
                    continue
            
            # BÖLÜM başlığı kontrolü
            bolum_match = re.match(r'^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ)\s+BÖLÜM\s*$', line, re.IGNORECASE)
            if bolum_match:
                self.current_bolum = line
                continue
            
            # BÖLÜM alt başlık
            if self.current_bolum and not line.startswith('Madde') and not self.current_bolum.endswith(line):
                if not re.match(r'^[a-z]', line):
                    self.current_bolum = f"{self.current_bolum} - {line}"
                    continue
            
            # Madde başlangıcı kontrolü
            madde_match = re.match(r'^Madde\s+(\d+(?:/[A-Z])?)\s*[–-]\s*(?:\(.*?\))?\s*(.*)$', line)
            if madde_match:
                # Önceki maddeyi kaydet
                if current_madde:
                    maddeler.append({
                        'madde_no': current_madde['madde_no'],
                        'madde_baslik': current_madde['madde_baslik'],
                        'kisim': current_madde['kisim'],
                        'bolum': current_madde['bolum'],
                        'icerik': '\n'.join(current_content).strip()
                    })
                
                # Yeni madde başlat
                madde_no = madde_match.group(1)
                madde_devam = madde_match.group(2).strip()
                
                current_madde = {
                    'madde_no': madde_no,
                    'madde_baslik': '',
                    'kisim': self.current_kisim,
                    'bolum': self.current_bolum,
                }
                current_content = []
                
                if madde_devam:
                    current_content.append(madde_devam)
                continue
            
            # iki nokta üst üste ile biten satırlar genelde başlıktır
            if current_madde and line.endswith(':') and not current_madde['madde_baslik']:
                current_madde['madde_baslik'] = line.rstrip(':')
                continue
            
            # Normal içerik satırı
            if current_madde:
                current_content.append(line)
        
        # Son maddeyi kaydet
        if current_madde and current_content:
            maddeler.append({
                'madde_no': current_madde['madde_no'],
                'madde_baslik': current_madde['madde_baslik'],
                'kisim': current_madde['kisim'],
                'bolum': current_madde['bolum'],
                'icerik': '\n'.join(current_content).strip()
            })
        
        return maddeler

    def create_chunks(self, madde: Dict) -> List[KanunChunk]:
        chunks = []
        
        alt_chunks = self.split_madde_content(madde['icerik'])
        
        for i, chunk_content in enumerate(alt_chunks):
            # Hiyerarşik içerik oluştur (genelgeler gibi)
            hierarchical_content = self._create_hierarchical_content(
                madde_no=madde['madde_no'],
                madde_baslik=madde['madde_baslik'],
                kisim=madde['kisim'],
                bolum=madde['bolum'],
                chunk_content=chunk_content
            )
            
            chunk_id = hashlib.md5(
                f"kanun-{madde['madde_no']}-{i}-{chunk_content[:50]}".encode()
            ).hexdigest()[:12]
            
            full_path = f"Noterlik Kanunu > Madde {madde['madde_no']}"
            if madde['madde_baslik']:
                full_path += f" ({madde['madde_baslik']})"
            
            chunk = KanunChunk(
                madde_no=madde['madde_no'],
                madde_baslik=madde['madde_baslik'],
                kisim=madde['kisim'],
                bolum=madde['bolum'],
                icerik=hierarchical_content,
                full_path=full_path,
                chunk_id=chunk_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_hierarchical_content(
        self,
        madde_no: str,
        madde_baslik: str,
        kisim: str,
        bolum: str,
        chunk_content: str
    ) -> str:
        hierarchical = f"NOTERLİK KANUNU (1512)\n"
        
        if kisim:
            hierarchical += f"{kisim}\n"
        
        if bolum:
            hierarchical += f"{bolum}\n"
        
        hierarchical += f"Madde {madde_no}"
        if madde_baslik:
            hierarchical += f" - {madde_baslik}"
        hierarchical += "\n"
        hierarchical += "---\n"
        hierarchical += chunk_content
        
        return hierarchical
    
    def split_madde_content(
        self,
        content: str,
        max_length: int = 1500,
        overlap: int = 200
    ) -> List[str]:
        """
        İçeriği chunklara böler, overlap ile context korunur
        """
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        
        # Numaralandırılmış bentlere göre böl (1., 2., 3. vb.)
        bent_pattern = r'(\d+\.\s+)'
        parts = re.split(bent_pattern, content)
        
        current_chunk = ""
        
        for i in range(0, len(parts)):
            part = parts[i]
            
            if len(current_chunk + part) <= max_length:
                current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Overlap
                    if len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + part
                    else:
                        current_chunk = part
                else:
                    # Part kendisi çok uzunsa, cümlelere göre böl
                    sentences = re.split(r'([.!?]\s+)', part)
                    temp_chunk = ""
                    
                    for j in range(0, len(sentences)):
                        sentence = sentences[j]
                        
                        if len(temp_chunk + sentence) <= max_length:
                            temp_chunk += sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sentence
                            else:
                                # Tek cümle bile çok uzunsa, zorla böl
                                chunks.append(sentence[:max_length].strip())
                                temp_chunk = sentence[max_length:]
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]

    def process_file(self, file_path: str) -> List[KanunChunk]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print("🔄 Noterlik Kanunu parse ediliyor...")
        self.maddeler = self.parse_kanun_text(text)
        print(f"✅ {len(self.maddeler)} madde bulundu")
        
        print("🔄 Chunklar oluşturuluyor...")
        for madde in self.maddeler:
            chunks = self.create_chunks(madde)
            self.chunks.extend(chunks)
        
        print(f"✅ {len(self.chunks)} chunk oluşturuldu")
        return self.chunks

    def export_for_rag(self, output_path: str):
        rag_data = []
        
        for chunk in self.chunks:
            rag_data.append({
                'id': chunk.chunk_id,
                'content': chunk.icerik,
                'metadata': {
                    'source_type': 'kanun',
                    'kanun_adi': 'Noterlik Kanunu',
                    'kanun_no': '1512',
                    'madde_no': chunk.madde_no,
                    'madde_baslik': chunk.madde_baslik,
                    'kisim': chunk.kisim,
                    'bolum': chunk.bolum,
                    'full_path': chunk.full_path,
                    'source': 'Noterlik Kanunu (1512)'
                }
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(rag_data)} chunk '{output_path}' dosyasına kaydedildi")

    def get_statistics(self):
        return {
            'toplam_madde': len(self.maddeler),
            'toplam_chunk': len(self.chunks),
            'ortalama_chunk_uzunlugu': (
                sum(len(chunk.icerik) for chunk in self.chunks) / len(self.chunks)
                if self.chunks else 0
            ),
            'kisimlar': list(set(m['kisim'] for m in self.maddeler if m['kisim'])),
            'madde_basina_chunk': (
                len(self.chunks) / len(self.maddeler) if self.maddeler else 0
            )
        }


if __name__ == "__main__":
    processor = NoterlikKanunuProcessor()
    
    chunks = processor.process_file("kanun_extracted.txt")
    
    processor.export_for_rag("noterlik_kanunu_rag.json")
    
    stats = processor.get_statistics()
    print("\n📊 İşlem İstatistikleri:")
    print(f"Toplam Madde: {stats['toplam_madde']}")
    print(f"Toplam Chunk: {stats['toplam_chunk']}")
    print(f"Ortalama Chunk Uzunluğu: {stats['ortalama_chunk_uzunlugu']:.0f} karakter")
    print(f"Madde Başına Chunk: {stats['madde_basina_chunk']:.2f}")
    print(f"\nBulunan Kısımlar: {len(stats['kisimlar'])}")
    
    print("\n📝 Örnek Chunklar:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n{i+1}. {chunk.full_path}")
        print(f"   ID: {chunk.chunk_id}")
        print(f"   Kısım: {chunk.kisim[:50]}...")
        print(f"   İçerik: {chunk.icerik[:150]}...")

