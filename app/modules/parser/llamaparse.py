import os
import json
import re
from llama_cloud import LlamaCloud
from app.core.config import settings

def parse_document_to_json(file_path: str, output_file: str = "raw_parsing_result.json") -> dict:
    """
    Fungsi untuk mem-parsing PDF menggunakan LlamaParse dan menyimpan hasilnya ke raw JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {file_path}")

    print(f"Mengupload file: {file_path}")
    client = LlamaCloud(api_key=settings.llamaparse_api_key)
    file_obj = client.files.create(file=file_path, purpose="parse")
    print(f"File terupload dengan ID: {file_obj.id}")

    print("Memulai parsing...")
    result = client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["text", "markdown", "items", "images_content_metadata"],
        output_options={
            "markdown": {
                "tables": {"output_tables_as_markdown": True},
                "inline_images": True
            },
            "images_to_save": ["embedded", "screenshot"]
        },
        processing_options={
            "ocr_parameters": {"languages": ["en", "id"]}
        }
    )
    print("Parsing selesai\n")
    # Konversi objek Pydantic ke dictionary
    try:
        raw_dict = result.model_dump()
    except AttributeError:
        try:
            raw_dict = result.dict()
        except AttributeError:
            raw_dict = result.to_dict()

    # Simpan ke file JSON
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(raw_dict, f, indent=2, ensure_ascii=False, default=str)
    # print(f"Raw JSON disimpan ke: {output_file}")

    return raw_dict

def reconstruct_content_list(data: dict, output_file: str = "formatted_parsing_result.json") -> list:
    content_list = []

    def get_first_sentence(text):
        if not text: return None
        text = text.strip()
        if not text: return None
        parts = re.split(r'(?<=[.?!])\s+', text, maxsplit=1)
        return parts[0].strip()

    def get_last_sentence(text):
        if not text: return None
        text = text.strip()
        if not text: return None
        parts = re.split(r'(?<=[.?!])\s+', text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts[-1] if parts else None

    # Pre-process images: Ambil hanya Chart yang valid (skip table & icon kecil)
    valid_images_by_page = {}
    for img in data.get("images_content_metadata", {}).get("images", []):
        filename = img.get("filename", "")
        category = img.get("category", "")
        bbox = img.get("bbox", {})
        
        if category != "layout": continue
        if bbox.get("w", 0) < 100 or bbox.get("h", 0) < 100: continue
        if "table" in filename.lower(): continue # Skip gambar tabel karena sudah ada di items
        
        is_chart = "chart" in filename.lower()
        try:
            parts = filename.split("_")
            page_num = int(parts[1])
        except: continue
            
        if page_num not in valid_images_by_page:
            valid_images_by_page[page_num] = []
            
        valid_images_by_page[page_num].append({
            "url": img.get("presigned_url"),
            "y": bbox.get("y", 0),
            "x": bbox.get("x", 0),  # Tambahan x untuk pencocokan bbox
            "is_chart": is_chart
        })

    # Dictionary untuk menyimpan data item ber-label "chart" agar bisa dicocokkan dengan gambar
    chart_items_by_page = {}

    # Iterasi setiap halaman
    for page_data in data.get("items", {}).get("pages", []):
        page_num = page_data.get("page_number", 0)
        page_idx = page_num - 1
        page_elements = []

        # Ekstrak items (Text, Heading, Table)
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
                label = bbox_list[0].get("label", "table") if bbox_list and isinstance(bbox_list, list) else "table"
                if label != "chart": # Skip teks chart, gunakan gambar
                    page_elements.append({
                        "type": "table",
                        "html": item.get("html", ""),
                        "y": y_coord
                    })
                else:
                    # Simpan data chart (bbox & md) untuk dicocokkan dengan gambar
                    if page_num not in chart_items_by_page:
                        chart_items_by_page[page_num] = []
                    chart_items_by_page[page_num].append({
                        "bbox": bbox_list[0],
                        "md": item.get("md", "")
                    })
            
        # Tambahkan gambar Chart
        if page_num in valid_images_by_page:
            for v_img in valid_images_by_page[page_num]:
                img_y = v_img["y"]
                img_x = v_img["x"]
                
                chart_md = None
                # Cocokkan item chart dengan gambar berdasarkan koordinat x dan y (toleransi 10px)
                if page_num in chart_items_by_page:
                    for chart_item in chart_items_by_page[page_num]:
                        c_bbox = chart_item["bbox"]
                        if abs(c_bbox.get("x", 0) - img_x) < 10 and abs(c_bbox.get("y", 0) - img_y) < 10:
                            chart_md = chart_item["md"]
                            break
                
                page_elements.append({
                    "type": "image",
                    "url": v_img["url"],
                    "y": img_y,
                    "is_chart": v_img["is_chart"],
                    "chart_data": chart_md
                })
                
        # Sortir berdasarkan Y (atas ke bawah)
        type_priority = {"image": 0, "text": 1, "table": 2}
        page_elements.sort(key=lambda x: (x["y"], type_priority.get(x["type"], 3)))
        
        # Proses elemen yang sudah diurutkan
        last_heading = None
        last_text = None
        
        for i, elem in enumerate(page_elements):
            if elem["type"] == "text":
                content_list.append({
                    "type": "text",
                    "text": elem["text"],
                    "page_idx": page_idx
                })
                if elem.get("is_heading"): last_heading = elem["text"]
                else: last_text = elem["text"]
                
            elif elem["type"] == "table":
                table_caption, table_footnote = [], []
                
                # Ambil kalimat terakhir dari teks di atas
                if i > 0:
                    prev_elem = page_elements[i-1]
                    if prev_elem["type"] == "text":
                        last_sent = get_last_sentence(prev_elem["text"])
                        if last_sent: table_caption.append(last_sent)
                    elif last_heading:
                        last_sent = get_last_sentence(last_heading)
                        if last_sent: table_caption.append(last_sent)
                elif last_heading:
                    last_sent = get_last_sentence(last_heading)
                    if last_sent: table_caption.append(last_sent)
                
                # Ambil kalimat pertama dari teks di bawah
                if i < len(page_elements) - 1:
                    next_elem = page_elements[i+1]
                    if next_elem["type"] == "text":
                        first_sent = get_first_sentence(next_elem["text"])
                        if first_sent: table_footnote.append(first_sent)
                
                if not table_caption and last_text and last_text != last_heading:
                    last_sent = get_last_sentence(last_text)
                    if last_sent: table_caption.append(last_sent)
                
                content_list.append({
                    "type": "table",
                    "table_body": elem["html"],  # Gunakan HTML
                    "table_caption": table_caption if table_caption else None,
                    "table_footnote": table_footnote if table_footnote else None,
                    "page_idx": page_idx
                })
                
            elif elem["type"] == "image":
                image_caption, image_footnote = [], []
                
                # Ambil kalimat terakhir dari teks di atas
                if i > 0:
                    prev_elem = page_elements[i-1]
                    if prev_elem["type"] == "text":
                        last_sent = get_last_sentence(prev_elem["text"])
                        if last_sent: image_caption.append(last_sent)
                    elif last_heading:
                        last_sent = get_last_sentence(last_heading)
                        if last_sent: image_caption.append(last_sent)
                elif last_heading:
                    last_sent = get_last_sentence(last_heading)
                    if last_sent: image_caption.append(last_sent)
                    
                if not image_caption and last_text:
                    last_sent = get_last_sentence(last_text)
                    if last_sent: image_caption.append(last_sent)
                
                # Ambil kalimat pertama dari teks di bawah
                if i < len(page_elements) - 1:
                    next_elem = page_elements[i+1]
                    if next_elem["type"] == "text":
                        first_sent = get_first_sentence(next_elem["text"])
                        if first_sent: image_caption.append(first_sent) # Masukkan ke caption untuk image
                
                # Sisipkan hasil items ber label "chart" (chart_data) ke caption
                if elem.get("chart_data"):
                    image_caption.append(elem["chart_data"])
                
                content_list.append({
                    "type": "image",
                    "img_path": elem["url"],
                    "image_caption": image_caption if image_caption else None,
                    "image_footnote": image_footnote if image_footnote else None,
                    "page_idx": page_idx
                })
        # # Simpan ke file JSON
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(content_list, f, indent=2, ensure_ascii=False, default=str)
        # print(f"Raw JSON disimpan ke: {output_file}")
    return content_list