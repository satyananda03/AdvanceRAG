import os
import json
import re
from llama_cloud import LlamaCloud
import os
from dotenv import load_dotenv

api_key = os.getenv("LLAMAPARSE_API_KEY")
client = LlamaCloud(api_key=api_key)

# Upload file
file_path = "test1.pdf"  # Ganti dengan file Anda
if not os.path.exists(file_path):
    print(f"❌ File tidak ditemukan: {file_path}")
    exit()

print(f"📤 Mengupload file: {file_path}")
file_obj = client.files.create(
    file=file_path,
    purpose="parse"
)
print(f"✅ File terupload dengan ID: {file_obj.id}")

# Parsing instructions
parsing_instruction = '''
You are a pure text extraction tool. 
Whenever you encounter an image, photo, graphic, or illustration, you MUST completely ignore it and output an empty string instead of describing it. 
DO NOT output any description, alt-text, or captions. Just output nothing.
'''

# Parse dengan konfigurasi lengkap
print("🔄 Memulai parsing...")
# Konfigurasi alternatif untuk mendapatkan text dan items
result = client.parsing.parse(
    file_id=file_obj.id,
    tier="agentic",
    version="latest",
    # Jangan gunakan parsing_instructions yang terlalu restriktif
    expand=["text", "markdown", "items", "images_content_metadata"],
    output_options={
        "markdown": {
            "tables": {
                "output_tables_as_markdown": True
            },
            "inline_images": True
        },
        "images_to_save": ["embedded", "screenshot"]
    },
    processing_options={
        "ocr_parameters": {
            "languages": ["en", "id"]
        }
    }
)

print("✅ Parsing selesai\n")

# === PRINT RAW JSON ===
print("=" * 80)
print("📄 RAW JSON OUTPUT")
print("=" * 80)

# Konversi objek Pydantic ke dictionary
try:
    # Metode 1: Menggunakan model_dump() (Pydantic v2)
    raw_dict = result.model_dump()
    print("✅ Menggunakan model_dump()")
except AttributeError:
    try:
        # Metode 2: Menggunakan dict() (Pydantic v1)
        raw_dict = result.dict()
        print("✅ Menggunakan dict()")
    except AttributeError:
        # Metode 3: Menggunakan to_dict()
        raw_dict = result.to_dict()
        print("✅ Menggunakan to_dict()")

# Cetak struktur utama
print(f"\n🔑 Key utama yang tersedia:")
for key in raw_dict.keys():
    value = raw_dict[key]
    if isinstance(value, list):
        print(f"   - {key}: List dengan {len(value)} item")
    elif isinstance(value, dict):
        print(f"   - {key}: Dict dengan {len(value)} key")
    elif value is None:
        print(f"   - {key}: None")
    else:
        print(f"   - {key}: {type(value).__name__}")

# Cetak raw JSON dengan indentasi
print("\n" + "=" * 80)
print("📋 FULL RAW JSON")
print("=" * 80)
print(json.dumps(raw_dict, indent=2, ensure_ascii=False, default=str))

# === SIMPAN KE FILE ===
output_file = "raw_parsing_result.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(raw_dict, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✅ Raw JSON disimpan ke: {output_file}")

# === ANALISIS STRUKTUR ===
print("\n" + "=" * 80)
print("🔍 ANALISIS STRUKTUR FIELD")
print("=" * 80)

# 1. Cek field markdown
if hasattr(result, 'markdown') and result.markdown:
    print(f"\n📝 MARKDOWN:")
    print(f"   - Type: {type(result.markdown)}")
    print(f"   - Attributes: {[attr for attr in dir(result.markdown) if not attr.startswith('_')]}")
    
    if hasattr(result.markdown, 'pages'):
        print(f"   - Jumlah halaman: {len(result.markdown.pages)}")
        for i, page in enumerate(result.markdown.pages[:3]):  # Print 3 halaman pertama
            print(f"\n   Halaman {i+1}:")
            print(f"   - Type: {type(page)}")
            print(f"   - Attributes: {[attr for attr in dir(page) if not attr.startswith('_')]}")
            
            # Print beberapa attribute penting
            for attr in ['page_number', 'text', 'markdown', 'images', 'items']:
                if hasattr(page, attr):
                    value = getattr(page, attr)
                    if value:
                        if isinstance(value, str):
                            print(f"   - {attr}: {len(value)} karakter")
                        elif isinstance(value, list):
                            print(f"   - {attr}: {len(value)} item")
                        else:
                            print(f"   - {attr}: {type(value).__name__}")
                    else:
                        print(f"   - {attr}: Kosong/None")

# 2. Cek field images_content_metadata
if hasattr(result, 'images_content_metadata') and result.images_content_metadata:
    print(f"\n🖼️ IMAGES_CONTENT_METADATA:")
    print(f"   - Type: {type(result.images_content_metadata)}")
    print(f"   - Attributes: {[attr for attr in dir(result.images_content_metadata) if not attr.startswith('_')]}")
    
    if hasattr(result.images_content_metadata, 'images'):
        images_list = result.images_content_metadata.images
        print(f"   - Jumlah gambar: {len(images_list)}")
        
        for i, img in enumerate(images_list[:5]):  # Print 5 gambar pertama
            print(f"\n   Gambar {i+1}:")
            print(f"   - Type: {type(img)}")
            print(f"   - Attributes: {[attr for attr in dir(img) if not attr.startswith('_')]}")
            
            # Print beberapa attribute penting
            for attr in ['filename', 'category', 'bbox', 'page_number', 'width', 'height', 'caption', 'alt_text']:
                if hasattr(img, attr):
                    value = getattr(img, attr)
                    print(f"   - {attr}: {value}")

# === DETAIL PER HALAMAN ===
print("\n" + "=" * 80)
print("📖 DETAIL PER HALAMAN")
print("=" * 80)

if hasattr(result, 'markdown') and hasattr(result.markdown, 'pages'):
    for i, page in enumerate(result.markdown.pages):
        print(f"\n--- Halaman {i+1} ---")
        
        # Cetak markdown
        if hasattr(page, 'markdown'):
            md = page.markdown
            print(f"   Markdown: {len(md) if md else 0} karakter")
            if md:
                # Cetak preview
                preview = md[:300] + "..." if len(md) > 300 else md
                print(f"   Preview: {preview}")
        
        # Cetak text
        if hasattr(page, 'text'):
            text = page.text
            print(f"   Text: {len(text) if text else 0} karakter")
        
        # Cetak images
        if hasattr(page, 'images'):
            images = page.images
            print(f"   Images: {len(images) if images else 0} item")
        
        # Cetak items
        if hasattr(page, 'items'):
            items = page.items
            print(f"   Items: {len(items) if items else 0} item")
            if items:
                for j, item in enumerate(items):
                    item_type = getattr(item, 'type', 'unknown') if hasattr(item, 'type') else 'unknown'
                    print(f"      Item {j+1}: type={item_type}")

print("\n" + "=" * 80)
print("✅ ANALISIS SELESAI")
print("=" * 80)