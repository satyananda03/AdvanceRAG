import io
import os
import json
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
import mimetypes
from pathlib import Path
import boto3
import fitz
from PIL import Image
from dotenv import load_dotenv
from app.core.logging import get_logger
logger = get_logger(__name__)
load_dotenv()

import io
import os
import json
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
import mimetypes
from pathlib import Path
import boto3
import fitz
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

LAYOUT_TEXT_TYPES = {
    "LAYOUT_TEXT",
    "LAYOUT_TITLE",
    "LAYOUT_SECTION_HEADER",
    "LAYOUT_LIST",
    "LAYOUT_KEY_VALUE",
}

IGNORED_LAYOUT_TYPES = {
    "LAYOUT_HEADER",
    "LAYOUT_FOOTER",
    "LAYOUT_PAGE_NUMBER",
}

MIN_FIGURE_WIDTH_PX = 400
MIN_FIGURE_HEIGHT_PX = 400
MIN_FIGURE_AREA_RATIO = 0.01

def generate_s3_key(file_path: str, prefix: str) -> str:
    """Generate S3 object key."""
    ext = Path(file_path).suffix.lower() or ".jpg"
    timestamp = time.strftime("%y%m%d%H%M%S")
    # short_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"{prefix}{timestamp}_{ext}"

def upload_to_s3(file_path: str, s3_key: str) -> str:
    """Upload local file ke S3 object storage"""
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("S3_REGION")
    s3_client = boto3.client('s3',
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        region_name=region,
    )
    s3_client.upload_file(
        file_path, 
        bucket, 
        s3_key, 
        ExtraArgs={'ContentType': content_type}
    )
    return f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"

# Render PDF to page images
def render_pdf_pages(pdf_path: str, images_dir: str, dpi: int = 200) -> list:
    """Render setiap halaman PDF menjadi PNG di disk dengan DPI tertentu."""
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    for page_idx, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        # Simpan page image ke disk untuk figure cropping nanti
        page_img_path = os.path.join(images_dir, f"page_{page_idx:04d}.png")
        pil_img.save(page_img_path, "PNG")
        pages.append({
            "page_idx": page_idx,
            "page_image_path": os.path.abspath(page_img_path),
            "png_bytes": png_bytes,
            "img_w": pil_img.size[0],
            "img_h": pil_img.size[1],
        })
    doc.close()
    return pages

# Process document dengan Textract
def _process_single_page(textract_client, page: dict) -> dict:
    """Proses satu halaman : panggil Textract lalu simpan hasil + metadata."""
    try:
        response = textract_client.analyze_document(
            Document={"Bytes": page["png_bytes"]},
            FeatureTypes=["LAYOUT", "TABLES"],
        )
        result = {
            "page_idx": page["page_idx"],
            "page_image_path": page["page_image_path"],
            "img_w": page["img_w"],
            "img_h": page["img_h"],
            "DocumentMetadata": response.get("DocumentMetadata"),
            "AnalyzeDocumentModelVersion": response.get("AnalyzeDocumentModelVersion"),
            "Blocks": response.get("Blocks", []),
            "error": None,
        }
    except Exception as e:
        result = {
            "page_idx": page["page_idx"],
            "page_image_path": page["page_image_path"],
            "img_w": page["img_w"],
            "img_h": page["img_h"],
            "DocumentMetadata": None,
            "AnalyzeDocumentModelVersion": None,
            "Blocks": [],
            "error": str(e),
        }
    finally:
        page["png_bytes"] = None
    return result

def build_block_map(blocks: list) -> dict:
    return {b["Id"]: b for b in blocks}

def get_children(block: dict, block_map: dict, rel_type: str = "CHILD") -> list:
    ids = []
    for rel in block.get("Relationships", []):
        if rel["Type"] == rel_type:
            ids.extend(rel["Ids"])
    return [block_map[i] for i in ids if i in block_map]

def get_plain_text(block: dict, block_map: dict) -> str:
    """Rekursive Loop sampai WORD/LINE, untuk merangkai teks."""
    if block["BlockType"] in ("WORD", "LINE"):
        return block.get("Text", "")
    parts = []
    for child in get_children(block, block_map):
        txt = get_plain_text(child, block_map)
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()

def get_word_blocks(block: dict, block_map: dict) -> list:
    """Rekursive Loop mengumpulkan semua block WORD dari suatu parent block."""
    if block["BlockType"] == "WORD":
        return [block]
    words = []
    for child in get_children(block, block_map):
        words.extend(get_word_blocks(child, block_map))
    return words

def is_vertical_text(block: dict, block_map: dict, angle_threshold: float = 45.0, ratio_threshold: float = 0.5) -> bool:
    words = get_word_blocks(block, block_map)
    if not words:
        return False
    rotated_count = sum(1 for w in words if abs(w.get("Geometry", {}).get("RotationAngle", 0.0)) > angle_threshold)
    return (rotated_count / len(words)) > ratio_threshold

def bbox_to_pixels(bbox: dict, img_w: int, img_h: int):
    left = bbox["Left"] * img_w
    top = bbox["Top"] * img_h
    width = bbox["Width"] * img_w
    height = bbox["Height"] * img_h
    return left, top, left + width, top + height

def bbox_iou(b1: dict, b2: dict) -> float:
    ax1, ay1 = b1["Left"], b1["Top"]
    ax2, ay2 = ax1 + b1["Width"], ay1 + b1["Height"]
    bx1, by1 = b2["Left"], b2["Top"]
    bx2, by2 = bx1 + b2["Width"], by1 + b2["Height"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = b1["Width"] * b1["Height"]
    area_b = b2["Width"] * b2["Height"]
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

def bbox_overlap_ratio(bbox1: dict, bbox2: dict) -> float:
    """Menghitung seberapa persen sebuah bbox1 tumpang tindih dengan bbox2."""
    ax1, ay1 = bbox1["Left"], bbox1["Top"]
    ax2, ay2 = ax1 + bbox1["Width"], ay1 + bbox1["Height"]
    bx1, by1 = bbox2["Left"], bbox2["Top"]
    bx2, by2 = bx1 + bbox2["Width"], by1 + bbox2["Height"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = bbox1["Width"] * bbox1["Height"]
    return inter_area / area1 if area1 > 0 else 0.0

def build_table_markdown(table_block: dict, block_map: dict) -> str:
    cell_blocks = [
        c for c in get_children(table_block, block_map)
        if c["BlockType"] in ("CELL", "MERGED_CELL")
    ]
    if not cell_blocks:
        return ""

    max_row = max(c.get("RowIndex", 1) + c.get("RowSpan", 1) - 1 for c in cell_blocks)
    max_col = max(c.get("ColumnIndex", 1) + c.get("ColumnSpan", 1) - 1 for c in cell_blocks)
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]

    for c in cell_blocks:
        r = c.get("RowIndex", 1) - 1
        col = c.get("ColumnIndex", 1) - 1
        if 0 <= r < max_row and 0 <= col < max_col:
            text = get_plain_text(c, block_map).replace("|", "/").replace("\n", " ")
            entity_types = c.get("EntityTypes", [])
            if "TABLE_SECTION_TITLE" in entity_types:
                # Penanda Bold untuk Table Section Title
                text = f"**{text}**"
            elif "COLUMN_HEADER" in entity_types:
                # Penanda Bold untuk Header Column
                text = f"**{text}**"
            grid[r][col] = text
    lines = []
    lines.append("| " + " | ".join(grid[0]) + " |")
    lines.append("|" + "|".join(["---"] * max_col) + "|")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def get_table_caption_footnote(table_block: dict, block_map: dict, header_bboxes: list = None, footer_bboxes: list = None) -> tuple:
    """Ekstrak caption & footnote dari TABLE block. Buang caption/footnote yang >50% berada di area header/footer."""
    captions, footnotes = [], []
    header_bboxes = header_bboxes or []
    footer_bboxes = footer_bboxes or []

    for rel in table_block.get("Relationships", []):
        if rel["Type"] == "TABLE_TITLE":
            for bid in rel["Ids"]:
                if bid in block_map:
                    cap_block = block_map[bid]
                    cap_bbox = cap_block["Geometry"]["BoundingBox"]
                    is_header = any(
                        bbox_overlap_ratio(cap_bbox, h_bbox) > 0.5
                        for h_bbox in header_bboxes
                    )
                    if not is_header:
                        captions.append(get_plain_text(cap_block, block_map))

        elif rel["Type"] == "TABLE_FOOTER":
            for bid in rel["Ids"]:
                if bid in block_map:
                    foot_block = block_map[bid]
                    foot_bbox = foot_block["Geometry"]["BoundingBox"]
                    is_footer = any(
                        bbox_overlap_ratio(foot_bbox, f_bbox) > 0.5
                        for f_bbox in footer_bboxes
                    )
                    if not is_footer:
                        footnotes.append(get_plain_text(foot_block, block_map))

    return captions, footnotes

def parse_document_to_json(file_path: str, output_file: str = "raw_output.json", region_name: str = None, max_workers: int = 5) -> dict:
    """Pipeline PDF -> raw Output JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {file_path}")

    output_file = os.path.abspath(output_file)
    tmp_dir = os.path.dirname(output_file)
    os.makedirs(tmp_dir, exist_ok=True)

    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    textract = boto3.client(
            "textract",
            aws_access_key_id=os.getenv("TEXTRACT_ACCESS_KEY_ID"),
            aws_secret_access_key= os.getenv("TEXTRACT_SECRET_ACCESS_KEY"),
            region_name= os.getenv("TEXTRACT_REGION"),
    )

    print(f"[Textract] Rendering PDF: {file_path}")
    pages = render_pdf_pages(file_path, images_dir)
    print(f"[Textract] {len(pages)} halaman dirender → {images_dir}")
    raw_pages = [None] * len(pages)

    if max_workers > 1:
        # Concurrent processing agar lebih cepat
        print(f"[Textract] Concurrent processing dengan {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_page, textract, page): page["page_idx"]
                for page in pages
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    raw_pages[idx] = future.result()
                except Exception as e:
                    raw_pages[idx] = {
                        "page_idx": idx,
                        "page_image_path": pages[idx]["page_image_path"],
                        "img_w": pages[idx]["img_w"],
                        "img_h": pages[idx]["img_h"],
                        "DocumentMetadata": None,
                        "AnalyzeDocumentModelVersion": None,
                        "Blocks": [],
                        "error": str(e),
                    }
                print(f"[Textract] Halaman {idx + 1}/{len(pages)} selesai" + (f" (ERROR: {raw_pages[idx]['error']})" if raw_pages[idx]["error"] else ""))
    else:
        # Sequential processing
        for page in pages:
            idx = page["page_idx"]
            print(f"[Textract] Memproses halaman {idx + 1}/{len(pages)}...")
            raw_pages[idx] = _process_single_page(textract, page)
            if raw_pages[idx]["error"]:
                print(f"[Textract] Halaman {idx + 1} error: {raw_pages[idx]['error']}")
            else:
                print(f"[Textract] Halaman {idx + 1} selesai")
    raw_data = {
        "source_pdf": os.path.abspath(file_path),
        "images_dir": images_dir,
        "pages": raw_pages,
    }
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(raw_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Textract] Raw JSON disimpan ke: {output_file}")
    return raw_data

# Format menjadi Content List
def reconstruct_content_list(data: dict, output_file: str = "formatted_output.json", min_figure_width_px: int = MIN_FIGURE_WIDTH_PX, min_figure_height_px: int = MIN_FIGURE_HEIGHT_PX, min_figure_area_ratio: float = MIN_FIGURE_AREA_RATIO) -> list:
    """Format raw JSON output Textract menjadi content list dari. Gambar otomatis di-upload ke S3"""
    # Auto-detect S3 config dari environment
    s3_bucket = os.getenv("S3_BUCKET")
    s3_enabled = bool(s3_bucket)
    s3_prefix = os.getenv("S3_PREFIX", "pdf-extractions/").rstrip("/") + "/"
    content_list = []
    images_dir = data.get("images_dir", "")
    for page_data in data.get("pages", []):
        page_idx = page_data["page_idx"]
        if page_data.get("error"):
            warnings.warn(f"Halaman {page_idx} di-skip (error: {page_data['error']})")
            continue
        blocks = page_data["Blocks"]
        if not blocks:
            continue
        page_image_path = page_data["page_image_path"]
        img_w = page_data["img_w"]
        img_h = page_data["img_h"]
        block_map = build_block_map(blocks)
        page_block = next((b for b in blocks if b["BlockType"] == "PAGE"), None)
        if page_block is None:
            continue
        ordered_children = get_children(page_block, block_map)
        header_bboxes = [b["Geometry"]["BoundingBox"] for b in ordered_children if b["BlockType"] == "LAYOUT_HEADER"]
        footer_bboxes = [b["Geometry"]["BoundingBox"] for b in ordered_children if b["BlockType"] == "LAYOUT_FOOTER"]
        table_blocks = [b for b in blocks if b["BlockType"] == "TABLE"]
        used_table_ids = set()
        pil_img = None
        idx = 0
        while idx < len(ordered_children):
            block = ordered_children[idx]
            btype = block["BlockType"]
            bbox = block["Geometry"]["BoundingBox"]
            if btype in IGNORED_LAYOUT_TYPES:
                idx += 1
                continue
            if btype in LAYOUT_TEXT_TYPES:
                if is_vertical_text(block, block_map):
                    idx += 1
                    continue
                text = get_plain_text(block, block_map)
                if text:
                    content_list.append({"type": "text", "text": text, "page_idx": page_idx})
            # Image Formatting
            elif btype == "LAYOUT_FIGURE":
                area_ratio = bbox["Width"] * bbox["Height"]
                left, top, right, bottom = bbox_to_pixels(bbox, img_w, img_h)
                crop_w_px, crop_h_px = right - left, bottom - top
                if (area_ratio < min_figure_area_ratio or crop_w_px < min_figure_width_px or crop_h_px < min_figure_height_px):
                    idx += 1
                    continue
                if pil_img is None:
                    pil_img = Image.open(page_image_path).convert("RGB")
                pad = 3
                crop_box = (
                    max(0, left - pad), max(0, top - pad),
                    min(img_w, right + pad), min(img_h, bottom + pad)
                )
                cropped = pil_img.crop(crop_box)
                # Simpan lokal sementara
                fname = f"figure_p{page_idx}_{uuid.uuid4().hex[:8]}.jpg"
                fpath = os.path.join(images_dir, fname) if images_dir else fname
                if images_dir:
                    os.makedirs(images_dir, exist_ok=True)
                cropped.save(fpath, "JPEG", quality=95)
                final_img_path = os.path.abspath(fpath)
                # Upload gambar ke S3
                if s3_enabled:
                    print("Upload to S3...")
                    try:
                        s3_key = generate_s3_key(final_img_path, s3_prefix)
                        final_img_path = upload_to_s3(final_img_path, s3_key)
                        print(f"Uploaded: {final_img_path}")
                    except Exception as e:
                        print(f"Failed to upload {fname} to S3: {e}")
                content_list.append({
                    "type": "image",
                    "img_path": final_img_path, 
                    "image_caption": None,
                    "image_footnote": None,
                    "page_idx": page_idx,
                })
            # Table Formatting
            elif btype == "LAYOUT_TABLE":
                best_table, best_iou = None, 0.0
                for t in table_blocks:
                    if t["Id"] in used_table_ids: continue
                    iou = bbox_iou(bbox, t["Geometry"]["BoundingBox"])
                    if iou > best_iou:
                        best_iou, best_table = iou, t
                if best_table is not None and best_iou > 0.3:
                    used_table_ids.add(best_table["Id"])
                    table_md = build_table_markdown(best_table, block_map)
                    captions, footnotes = get_table_caption_footnote(best_table, block_map, header_bboxes, footer_bboxes)
                else:
                    table_md = get_plain_text(block, block_map)
                    captions, footnotes = [], []
                content_list.append({
                    "type": "table",
                    "table_body": table_md,
                    "table_caption": captions if captions else None,
                    "table_footnote": footnotes if footnotes else None,
                    "page_idx": page_idx,
                })
            idx += 1
    if output_file is None:
        if images_dir:
            tmp_dir = os.path.dirname(images_dir)
            output_file = os.path.join(tmp_dir, "formatted_parsing_result.json")
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(content_list, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Textract] Formatted JSON disimpan ke: {output_file}")
    return content_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Amazon Textract PDF Parser")
    parser.add_argument("--input", required=True, dest="pdf_path", help="Path file PDF input")
    args = parser.parse_args()
    raw_data = parse_document_to_json(file_path=args.pdf_path)
    content_list = reconstruct_content_list(raw_data)
    print(f"\nSelesai. {len(content_list)} elemen diekstrak.")