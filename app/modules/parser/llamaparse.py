import os
import json
import re
from llama_cloud import LlamaCloud
from app.core.config import settings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTENT_DIR = os.path.join(SCRIPT_DIR, "debug")
os.makedirs(PERSISTENT_DIR, exist_ok=True)

DEFAULT_RAW_OUTPUT = os.path.join(PERSISTENT_DIR, "raw_parsing_result.json")
DEFAULT_FORMATTED_OUTPUT = os.path.join(PERSISTENT_DIR, "formatted_parsing_result.json")


def parse_document_to_json(file_path: str, output_file: str = DEFAULT_RAW_OUTPUT) -> dict:
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

    try:
        raw_dict = result.model_dump()
    except AttributeError:
        try:
            raw_dict = result.dict()
        except AttributeError:
            raw_dict = result.to_dict()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(raw_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"Raw JSON disimpan ke: {output_file}")

    return raw_dict


# ============================================================
# Helper functions for column-aware reading order sorting
# ============================================================

def _get_element_bbox(item: dict) -> dict | None:
    """
    Extract bbox info from a page item.

    - Sorts bboxes by (x, y) to find the "first" bbox (leftmost, topmost).
    - Uses first bbox for starting position (x, y) and column classification.
    - Uses all bboxes for overall extent (x_end, y_end).

    Returns dict with keys: x, y, w, h, x_end, y_end, first_bbox
    Returns None if no valid bbox found.
    """
    bbox_list = item.get("bbox", [])
    if not bbox_list or not isinstance(bbox_list, list):
        return None

    valid_bboxes = [b for b in bbox_list if isinstance(b, dict) and "x" in b and "y" in b]
    if not valid_bboxes:
        return None

    # Sort by (x, y) to get reading order (left-to-right, top-to-bottom)
    valid_bboxes.sort(key=lambda b: (b["x"], b["y"]))

    # Use first bbox for starting position
    first = valid_bboxes[0]
    x_start = first["x"]
    y_start = first["y"]
    first_w = first.get("w", 0)
    first_h = first.get("h", 0)

    # Use all bboxes for extent
    x_end = max(b["x"] + b.get("w", 0) for b in valid_bboxes)
    y_end = max(b["y"] + b.get("h", 0) for b in valid_bboxes)

    return {
        "x": x_start,
        "y": y_start,
        "w": first_w,
        "h": first_h,
        "x_end": x_start + first_w,
        "y_end": y_end,
        "first_bbox": first
    }


def _detect_columns(elements_with_bbox: list, page_width: float) -> list:
    """
    Detect column x-ranges from narrow elements.

    A "narrow" element is one whose width < 60% of page width.
    Columns are detected by clustering x_start positions.

    Returns list of {x_start, x_end} for each column.
    Returns empty list if single-column layout detected.
    """
    # Collect x_starts of narrow elements
    narrow_x_starts = []
    for _, bbox in elements_with_bbox:
        if bbox and bbox["w"] < page_width * 0.6:
            narrow_x_starts.append(bbox["x"])

    if not narrow_x_starts:
        return []

    narrow_x_starts.sort()

    # Cluster by x_start proximity (within 50 points = same column)
    column_starts = []
    for x in narrow_x_starts:
        if column_starts and x - column_starts[-1] < 50:
            continue
        column_starts.append(x)

    if len(column_starts) < 2:
        return []  # Single column

    # For each column, compute x_end from its elements
    columns = []
    for col_x in column_starts:
        col_elems = [
            bbox for _, bbox in elements_with_bbox
            if bbox and bbox["w"] < page_width * 0.6
            and abs(bbox["x"] - col_x) < 50
        ]
        if col_elems:
            x_end = max(b["x_end"] for b in col_elems)
            columns.append({"x_start": col_x, "x_end": x_end})

    return columns


def _classify_element_column(bbox: dict | None, columns: list) -> int:
    """
    Determine which column an element belongs to.

    Uses first bbox for classification.
    Returns col_idx (0-based) or -1 for spanning (overlaps multiple columns).
    """
    if not bbox or not columns:
        return -1

    first = bbox.get("first_bbox")
    if not first:
        return -1

    fb_x = first["x"]
    fb_x_end = first["x"] + first.get("w", 0)

    overlapping_cols = []
    for i, col in enumerate(columns):
        overlap = min(fb_x_end, col["x_end"]) - max(fb_x, col["x_start"])
        if overlap > 0:
            overlapping_cols.append(i)

    if len(overlapping_cols) >= 2:
        return -1  # Spans multiple columns
    elif len(overlapping_cols) == 1:
        return overlapping_cols[0]
    else:
        # No overlap - assign to nearest column
        best_col = 0
        best_dist = float('inf')
        for i, col in enumerate(columns):
            dist = abs(fb_x - col["x_start"])
            if dist < best_dist:
                best_dist = dist
                best_col = i
        return best_col


def _sort_elements_by_reading_order(page_elements: list, page_width: float) -> list:
    """
    Sort page elements by reading order, considering multi-column layout.

    Algorithm:
    1. Detect columns from narrow elements.
    2. If single column, sort by (y, type_priority, x).
    3. If multi-column:
       a. Classify each element as narrow (single column) or spanning.
       b. Sort all elements by y.
       c. Walk through in y order:
          - Spanning elements flush the current narrow band and are added directly.
          - Narrow elements accumulate in the current band.
       d. Within each narrow band, sort by (col_idx, y, type_priority, x).

    This produces column-major reading order: left column top-to-bottom,
    then right column top-to-bottom, etc.
    """
    if not page_elements:
        return page_elements

    TYPE_PRIORITY = {"image": 0, "text": 1, "table": 2}

    # Build list with bbox info
    elements_with_bbox = []
    for elem in page_elements:
        bbox = elem.get("bbox_info")
        elements_with_bbox.append((elem, bbox))

    # Detect columns
    columns = _detect_columns(elements_with_bbox, page_width)

    if len(columns) < 2:
        # Single column - sort by y, then type_priority, then x
        elements_with_bbox.sort(key=lambda x: (
            x[1]["y"] if x[1] else 99999,
            TYPE_PRIORITY.get(x[0]["type"], 3),
            x[1]["x"] if x[1] else 0
        ))
        return [e for e, _ in elements_with_bbox]

    # Multi-column: classify each element
    annotated = []
    for elem, bbox in elements_with_bbox:
        col_idx = _classify_element_column(bbox, columns)
        annotated.append({
            "elem": elem,
            "bbox": bbox,
            "col_idx": col_idx,
            "is_spanning": col_idx == -1,
            "y": bbox["y"] if bbox else 99999,
            "x": bbox["x"] if bbox else 0,
            "type": elem.get("type", "text")
        })

    # Sort by y, then type_priority (for elements at same y)
    annotated.sort(key=lambda x: (
        x["y"],
        TYPE_PRIORITY.get(x["type"], 3)
    ))

    # Walk through and group into bands
    result = []
    current_band = []

    for ann in annotated:
        if ann["is_spanning"]:
            # Flush current narrow band
            if current_band:
                current_band.sort(key=lambda x: (
                    x["col_idx"],
                    x["y"],
                    TYPE_PRIORITY.get(x["type"], 3),
                    x["x"]
                ))
                result.extend([a["elem"] for a in current_band])
                current_band = []
            # Add spanning element directly
            result.append(ann["elem"])
        else:
            current_band.append(ann)

    # Flush remaining band
    if current_band:
        current_band.sort(key=lambda x: (
            x["col_idx"],
            x["y"],
            TYPE_PRIORITY.get(x["type"], 3),
            x["x"]
        ))
        result.extend([a["elem"] for a in current_band])

    return result


# ============================================================
# Main reconstruction function
# ============================================================

def reconstruct_content_list(data: dict, output_file: str = DEFAULT_FORMATTED_OUTPUT) -> list:
    """
    Reconstruct content list from LlamaParse raw JSON.

    Handles multi-column layouts by detecting columns and sorting elements
    in column-major reading order (left column top-to-bottom, then right column).
    """
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

    # Pre-process images
    valid_images_by_page = {}
    for img in data.get("images_content_metadata", {}).get("images", []):
        filename = img.get("filename", "")
        category = img.get("category", "")
        bbox = img.get("bbox", {})

        if category != "layout":
            continue
        if bbox.get("w", 0) < 100 or bbox.get("h", 0) < 100:
            continue
        if "table" in filename.lower():
            continue

        is_chart = "chart" in filename.lower()
        try:
            parts = filename.split("_")
            page_num = int(parts[1])
        except (ValueError, IndexError):
            continue

        if page_num not in valid_images_by_page:
            valid_images_by_page[page_num] = []

        valid_images_by_page[page_num].append({
            "url": img.get("presigned_url"),
            "y": bbox.get("y", 0),
            "x": bbox.get("x", 0),
            "w": bbox.get("w", 0),
            "h": bbox.get("h", 0),
            "is_chart": is_chart
        })

    chart_items_by_page = {}

    for page_data in data.get("items", {}).get("pages", []):
        page_num = page_data.get("page_number", 0)
        page_idx = page_num - 1
        page_width = page_data.get("page_width", 595.0)
        page_elements = []

        for item in page_data.get("items", []):
            item_type = item.get("type")

            # Skip header and footer items
            if item_type in ["header", "footer"]:
                continue

            # Compute bbox info for column-aware sorting
            bbox_info = _get_element_bbox(item)
            y_coord = bbox_info["y"] if bbox_info else 99999

            if item_type in ["text", "heading"]:
                page_elements.append({
                    "type": "text",
                    "text": item.get("md", ""),
                    "y": y_coord,
                    "bbox_info": bbox_info,
                    "is_heading": item_type == "heading"
                })

            # ✅ Handle list type
            elif item_type == "list":
                list_md = item.get("md", "")
                if list_md:
                    if (page_elements
                        and page_elements[-1]["type"] == "text"
                        and not page_elements[-1].get("is_heading")):
                        page_elements[-1]["text"] += "\n" + list_md
                        # Update bbox_info if current item has one and previous doesn't
                        if bbox_info and not page_elements[-1].get("bbox_info"):
                            page_elements[-1]["bbox_info"] = bbox_info
                            page_elements[-1]["y"] = y_coord
                    else:
                        page_elements.append({
                            "type": "text",
                            "text": list_md,
                            "y": y_coord,
                            "bbox_info": bbox_info,
                            "is_heading": False
                        })

            elif item_type == "table":
                bbox_list = item.get("bbox", [])
                label = "table"
                if bbox_list and isinstance(bbox_list, list) and isinstance(bbox_list[0], dict):
                    label = bbox_list[0].get("label", "table")

                if label != "chart":
                    page_elements.append({
                        "type": "table",
                        "html": item.get("html", ""),
                        "y": y_coord,
                        "bbox_info": bbox_info
                    })
                else:
                    # Chart table - store for matching with chart image
                    if page_num not in chart_items_by_page:
                        chart_items_by_page[page_num] = []
                    chart_bbox = {}
                    if bbox_list and isinstance(bbox_list, list) and isinstance(bbox_list[0], dict):
                        chart_bbox = bbox_list[0]
                    chart_items_by_page[page_num].append({
                        "bbox": chart_bbox,
                        "md": item.get("md", "")
                    })

        # Add chart images to page elements
        if page_num in valid_images_by_page:
            for v_img in valid_images_by_page[page_num]:
                img_y = v_img["y"]
                img_x = v_img["x"]
                img_w = v_img["w"]
                img_h = v_img["h"]

                # Match with chart table data if available
                chart_md = None
                if page_num in chart_items_by_page:
                    for chart_item in chart_items_by_page[page_num]:
                        c_bbox = chart_item["bbox"]
                        if (abs(c_bbox.get("x", 0) - img_x) < 10
                            and abs(c_bbox.get("y", 0) - img_y) < 10):
                            chart_md = chart_item["md"]
                            break

                # Build bbox_info for image
                img_bbox_info = {
                    "x": img_x,
                    "y": img_y,
                    "w": img_w,
                    "h": img_h,
                    "x_end": img_x + img_w,
                    "y_end": img_y + img_h,
                    "first_bbox": {"x": img_x, "y": img_y, "w": img_w, "h": img_h}
                }

                page_elements.append({
                    "type": "image",
                    "url": v_img["url"],
                    "y": img_y,
                    "bbox_info": img_bbox_info,
                    "is_chart": v_img["is_chart"],
                    "chart_data": chart_md
                })

        # ✅ Sort elements by reading order (column-aware)
        page_elements = _sort_elements_by_reading_order(page_elements, page_width)

        # Build content list from sorted elements
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
                table_caption, table_footnote = [], []

                # Caption: last sentence of previous text element
                if i > 0:
                    prev_elem = page_elements[i - 1]
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

                # Footnote: first sentence of next text element
                if i < len(page_elements) - 1:
                    next_elem = page_elements[i + 1]
                    if next_elem["type"] == "text":
                        first_sent = get_first_sentence(next_elem["text"])
                        if first_sent:
                            table_footnote.append(first_sent)

                # Fallback caption from last_text
                if not table_caption and last_text and last_text != last_heading:
                    last_sent = get_last_sentence(last_text)
                    if last_sent:
                        table_caption.append(last_sent)

                content_list.append({
                    "type": "table",
                    "table_body": elem["html"],
                    "table_caption": table_caption if table_caption else None,
                    "table_footnote": table_footnote if table_footnote else None,
                    "page_idx": page_idx
                })

            elif elem["type"] == "image":
                image_caption, image_footnote = [], []

                # Caption: last sentence of previous text element
                if i > 0:
                    prev_elem = page_elements[i - 1]
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

                # Fallback caption from last_text
                if not image_caption and last_text:
                    last_sent = get_last_sentence(last_text)
                    if last_sent:
                        image_caption.append(last_sent)

                # Also check next element for caption
                if i < len(page_elements) - 1:
                    next_elem = page_elements[i + 1]
                    if next_elem["type"] == "text":
                        first_sent = get_first_sentence(next_elem["text"])
                        if first_sent:
                            image_caption.append(first_sent)

                # Add chart data if available
                if elem.get("chart_data"):
                    image_caption.append(elem["chart_data"])

                content_list.append({
                    "type": "image",
                    "img_path": elem["url"],
                    "image_caption": image_caption if image_caption else None,
                    "image_footnote": image_footnote if image_footnote else None,
                    "page_idx": page_idx
                })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content_list, f, indent=2, ensure_ascii=False, default=str)
    print(f"Formatted JSON disimpan ke: {output_file}")

    return content_list