import json
import re

# 1. Baca file raw JSON dari LlamaParse
with open("raw_parsing_result.json", "r") as f:
    data = json.load(f)

content_list = []

# ==========================================
# HELPER FUNCTIONS: AMBIL KALIMAT PERTAMA & TERAKHIR
# ==========================================

def get_first_sentence(text):
    """Ambil kalimat pertama dari sebuah teks (sampai titik/tanda baca akhir)."""
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    
    # Split berdasarkan titik, tanda tanya, atau tanda seru yang diikuti spasi/akhir
    parts = re.split(r'(?<=[.?!])\s+', text, maxsplit=1)
    first_sentence = parts[0].strip()
    
    return first_sentence

def get_last_sentence(text):
    """Ambil kalimat terakhir dari sebuah teks."""
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    
    # Split semua kalimat
    parts = re.split(r'(?<=[.?!])\s+', text)
    # Filter out empty strings
    parts = [p.strip() for p in parts if p.strip()]
    
    if not parts:
        return None
        
    return parts[-1]

# 2. Pre-process images_content_metadata (HANYA AMBIL CHART & IMAGE BERMAKNA)
valid_images_by_page = {}

for img in data.get("images_content_metadata", {}).get("images", []):
    filename = img.get("filename", "")
    category = img.get("category", "")
    bbox = img.get("bbox", {})
    
    if category != "layout":
        continue
        
    w = bbox.get("w", 0)
    h = bbox.get("h", 0)
    y = bbox.get("y", 0)
    
    # Filter icon kecil
    if w < 100 or h < 100:
        continue
        
    # DEDUPLICATION & STRATEGY:
    # 1. Skip gambar "table" (sudah diekstrak markdown di items)
    if "table" in filename.lower():
        continue
        
    # 2. Ambil gambar "chart" (akan diproses VLM)
    is_chart = "chart" in filename.lower()
    
    # Ekstrak nomor halaman
    try:
        parts = filename.split("_")
        page_num = int(parts[1])
    except (IndexError, ValueError):
        continue
        
    if page_num not in valid_images_by_page:
        valid_images_by_page[page_num] = []
        
    valid_images_by_page[page_num].append({
        "url": img.get("presigned_url"),
        "y": y,
        "filename": filename,
        "is_chart": is_chart
    })

# 3. Iterasi setiap halaman untuk rekonstruksi berbasis koordinat (Reading Order)
for page_data in data.get("items", {}).get("pages", []):
    page_num = page_data.get("page_number", 0)
    page_idx = page_num - 1  # RAGAnything menggunakan 0-indexed
    
    page_elements = []
    
    # Ekstrak dari items (Text, Heading, Table)
    for item in page_data.get("items", []):
        item_type = item.get("type")
        bbox_list = item.get("bbox", [])
        
        y_coord = 9999
        if bbox_list and isinstance(bbox_list, list):
            for b in bbox_list:
                if isinstance(b, dict) and "y" in b:
                    y_coord = min(y_coord, b["y"])
        
        if item_type in ["text", "heading"]:
            page_elements.append({
                "type": "text",
                "text": item.get("md", ""),
                "y": y_coord,
                "is_heading": item_type == "heading"
            })
        elif item_type == "table":
            # STRATEGY: Cek apakah ini chart yang di-force jadi table oleh LlamaParse
            label = bbox_list[0].get("label", "table") if bbox_list and isinstance(bbox_list, list) else "table"
            
            if label == "chart":
                # Abaikan ekstraksi teks chart karena tidak akurat, akan pakai gambar chart (VLM)
                continue
            else:
                # Ini tabel asli, ambil HTML-nya
                page_elements.append({
                    "type": "table",
                    "text": item.get("md", ""),  # Simpan md sebagai fallback
                    "html": item.get("html", ""), # Ambil format HTML
                    "y": y_coord
                })
            
    # Tambahkan gambar CHART dari metadata ke list elemen
    if page_num in valid_images_by_page:
        for v_img in valid_images_by_page[page_num]:
            page_elements.append({
                "type": "image",
                "url": v_img["url"],
                "y": v_img["y"],
                "is_chart": v_img["is_chart"]
            })
            
    # SORTING: Urutkan elemen berdasarkan Y (atas ke bawah)
    type_priority = {"image": 0, "text": 1, "table": 2}
    page_elements.sort(key=lambda x: (x["y"], type_priority.get(x["type"], 3)))
    
    # 4. Proses elemen yang sudah diurutkan menjadi format RAGAnything
    last_heading = None
    last_text = None
    
    for i, elem in enumerate(page_elements):
        if elem["type"] == "text":
            content_list.append({
                "type": "text",
                "text": elem["text"],
                "page_idx": page_idx
            })
            if elem.get("is_heading"):
                last_heading = elem["text"]
            else:
                last_text = elem["text"]
                
        elif elem["type"] == "table":
            table_caption = []
            table_footnote = []
            
            # ALGORITMA AMBIL TEKS ATAS (MENGAPIT ATAS) -> KALIMAT TERAKHIR
            if i > 0:
                prev_elem = page_elements[i-1]
                if prev_elem["type"] == "text":
                    last_sent = get_last_sentence(prev_elem["text"])
                    if last_sent:
                        table_caption.append(last_sent)
                elif last_heading:
                    last_sent = get_last_sentence(last_heading)
                    if last_sent:
                        table_caption.append(last_sent)
            elif last_heading:
                last_sent = get_last_sentence(last_heading)
                if last_sent:
                    table_caption.append(last_sent)
            
            # ALGORITMA AMBIL TEKS BAWAH (MENGAPIT BAWAH) -> KALIMAT PERTAMA
            if i < len(page_elements) - 1:
                next_elem = page_elements[i+1]
                if next_elem["type"] == "text":
                    first_sent = get_first_sentence(next_elem["text"])
                    if first_sent:
                        table_footnote.append(first_sent)
            
            # Fallback jika tidak ada pengapit sama sekali
            if not table_caption and last_text and last_text != last_heading:
                last_sent = get_last_sentence(last_text)
                if last_sent:
                    table_caption.append(last_sent)
            
            content_list.append({
                "type": "table",
                "table_body": elem["html"],  # GUNAKAN HTML DI SINI
                "table_caption": table_caption if table_caption else None,
                "table_footnote": table_footnote if table_footnote else None,
                "page_idx": page_idx
            })
            
        elif elem["type"] == "image":
            image_caption = []
            image_footnote = []
            
            # ALGORITMA AMBIL TEKS ATAS (MENGAPIT ATAS) -> KALIMAT TERAKHIR
            if i > 0:
                prev_elem = page_elements[i-1]
                if prev_elem["type"] == "text":
                    last_sent = get_last_sentence(prev_elem["text"])
                    if last_sent:
                        image_caption.append(last_sent)
                elif last_heading:
                    last_sent = get_last_sentence(last_heading)
                    if last_sent:
                        image_caption.append(last_sent)
            elif last_heading:
                last_sent = get_last_sentence(last_heading)
                if last_sent:
                    image_caption.append(last_sent)
                    
            if not image_caption and last_text:
                last_sent = get_last_sentence(last_text)
                if last_sent:
                    image_caption.append(last_sent)
            
            # ALGORITMA AMBIL TEKS BAWAH (MENGAPIT BAWAH) -> KALIMAT PERTAMA
            if i < len(page_elements) - 1:
                next_elem = page_elements[i+1]
                if next_elem["type"] == "text":
                    first_sent = get_first_sentence(next_elem["text"])
                    if first_sent:
                        # Untuk image, teks di bawah biasanya caption (Figure 1: ...)
                        # jadi masukkan ke caption
                        image_caption.append(first_sent)
            
            content_list.append({
                "type": "image",
                "img_path": elem["url"],
                "image_caption": image_caption if image_caption else None,
                "image_footnote": image_footnote if image_footnote else None,
                "page_idx": page_idx
            })

# 5. Cek hasil di terminal
print(f"Total item yang berhasil direkonstruksi: {len(content_list)}")
print("\nContoh 3 item pertama:")
for item in content_list[:3]:
    print(json.dumps(item, indent=2, ensure_ascii=False))

# 6. Simpan hasil ke file JSON
output_filename = "content_list_robust_vlm.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(content_list, f, indent=2, ensure_ascii=False)

print(f"\n✅ Hasil rekonstruksi robust berhasil disimpan ke file: {output_filename}")