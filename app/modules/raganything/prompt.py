"""
Prompt templates for multimodal content processing

Contains all prompt templates used in modal processors for analyzing
different types of content (images, tables, equations, etc.)
"""

from __future__ import annotations
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any


class PromptRegistry:
    """Stable prompt container with atomic snapshot swapping.

    Readers keep a reference to this object, while language switches replace the
    underlying prompt dictionary in one step via :meth:`swap`.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def swap(self, prompts: dict[str, Any]) -> None:
        """Atomically replace the active prompt snapshot."""
        self._data = dict(prompts)

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the active prompt set."""
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def items(self) -> ItemsView[str, Any]:
        return self._data.items()

    def values(self) -> ValuesView[Any]:
        return self._data.values()

    def __repr__(self) -> str:
        return f"PromptRegistry({self._data!r})"

PROMPTS = PromptRegistry()

# System prompts for different analysis types
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "ROLE : You are an expert image analyst. You ALWAYS respond with a valid JSON object only. "
    "Never include text outside the JSON. Never use markdown code blocks. "
    "Your response MUST contain 'detailed_description' and 'entity_info' keys."
)
PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "ROLE : You are an expert image analyst. You ALWAYS respond with a valid JSON object only. "
    "Never include text outside the JSON. Never use markdown code blocks. "
    "Your response MUST contain 'detailed_description' and 'entity_info' keys."
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "ROLE : You are an expert document analyst and legal/policy researcher. Provide a highly detailed and comprehensive analysis of the table content. "
    "LANGUAGE : BAHASA INDONESIA"
    "CRITICAL INSTRUCTION : For qualitative or text-heavy tables (such as government matrices, criteria, or regulations), YOU MUST extract and comprehensively explain the descriptions, criteria, administrative requirements, and institutional roles found in the cells. "
    "MANDATORY EXTRACTION : You are strictly required to identify and extract all key data points. This includes exact scores, thresholds, maturity levels, legal terms, specific clauses, and exact names of institutions or functions. "
    "Do not merely summarize or gloss over long texts. Capture the exact nuances, policy narratives, and structural relationships, as these detailed explanations and key data points are the core insights."
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "ROLE : You are an expert mathematician. Provide detailed mathematical analysis."
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "ROLE : You are an expert content analyst specializing in {content_type} content."
)

# Image analysis prompt template
PROMPTS["vision_prompt"] = """
You MUST respond with ONLY a valid JSON object. Do NOT include any explanation, markdown, preamble, or text outside the JSON. Do NOT wrap the JSON in code blocks.

<instructions>
1. CONTENT RECOGNITION & ADAPTABILITY
   Examine the image carefully. Identify its implicit type (e.g., Photo, Chart, Screenshot, Diagram, Flowchart, Infographic) and adapt your analysis:
   - For Charts/Diagrams: Identify axes, labels, legends, and extract specific quantitative data points/trends.
   - For Screenshots/Interfaces: Identify layout, main functional components, and visible text.
   - For Photos/Illustrations: Identify the primary subject, scene, spatial layout, and relationships between elements.
   - For Text/Tables: Quote short important text verbatim; summarize long blocks. Always use specific names instead of generic pronouns.

2. USE OF ADDITIONAL CONTEXT
   You are provided with contextual metadata (Section Path, Image Path, Captions, Footnotes).
   - Use this context to disambiguate acronyms, abbreviations, units, and entities.
   - STRICT RULE: The IMAGE ITSELF takes absolute priority. Describe only what is visually verifiable. Do NOT use the context to invent or assume visual elements that are not present in the image.

3. DETAILED DESCRIPTION (`detailed_description`)
   - Write a comprehensive, natural prose description (do NOT use bullet points).
   - Cover overall composition, visual style, colors, lighting, actions, and meaningful relationships between elements.
   - This field MUST NOT be empty or null.

4. ENTITY INFO (`entity_info`)
   - `entity_name`: Generate a clean, semantic, and distinctive title (3–8 words, snake_case preferred) representing the image content. Do NOT return raw file names or generic figure numbers (like `figure_30_1`) unless that is the literal title written on the image.
   - `entity_type`: MUST be strictly hardcoded as "image".
   - `summary`: Write a concise summary of the image content and its core significance (MAX 100 words). This field MUST NOT be empty or null.

5. JSON OUTPUT & ESCAPING RULES
   - Output EXACTLY ONE valid JSON object matching the structure below.
   - All string values must be properly escaped JSON strings: escape double quotes as \\", backslashes as \\\\, and newlines as \\n.
</instructions>

<context>
- Section Path: {section_path}
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}
</context>

<output>
Follow this output format exactly :
{{
    "detailed_description": "<comprehensive visual description following the instructions above, natural prose only>",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "<concise summary of image and significance, max 100 words>"
    }}
}}

Output:
</output>"""

# Image analysis prompt with context support
PROMPTS["vision_prompt_with_context"] = """
You MUST respond with ONLY a valid JSON object. Do NOT include any explanation, markdown, preamble, or text outside the JSON. Do NOT wrap the JSON in code blocks.
Your response MUST contain EXACTLY these two top-level keys: "detailed_description" and "entity_info". Any response missing either key is invalid and will cause a processing error.

Required JSON structure:
{{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements and how they relate to the surrounding context
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Reference connections to the surrounding content when relevant
    - Always use specific names instead of pronouns
    - This field MUST NOT be empty or null",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content, its significance, and relationship to surrounding content (max 100 words). This field MUST NOT be empty or null."
    }}
}}

Context from surrounding content:
{context}

Document structure:
- Section Path: {section_path}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

STRICT RULES:
1. Output ONLY the JSON object — no other text before or after.
2. Both "detailed_description" and "entity_info" are REQUIRED and MUST have non-empty values.
3. "entity_info" MUST contain all three keys: "entity_name", "entity_type", and "summary".
4. Use a semantic entity_name; do not return file names or figure numbers such as figure_30_1 unless they are the actual title.
5. If the image is unclear, still provide your best analysis — do NOT return an empty or partial response."""

# Image analysis prompt with text fallback
PROMPTS["text_prompt"] = """
Based on the following image information, provide analysis:

Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

{vision_prompt}
"""

# Table analysis prompt template
PROMPTS["table_prompt"] = """
Please analyze this table content and provide a JSON response with the following structure:
LANGUAGE : Bahasa Indonesia
{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    - Legal, policy, or governmental data points (e.g., regulations, articles, laws, official decrees)
    - Compliance implications, sanctions, or administrative requirements (if applicable)
    - Roles, jurisdictions, or hierarchy of government entities/institutions mentioned
    Always use specific names, values, and legal citations (Pasal, UU, Peraturan) instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "Concise summary of the table's purpose, key findings, and its regulatory/governmental context (max 200 words)"
    }}
}}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights, relationships, and the regulatory or administrative context from the tabular data.
"""

# Table analysis prompt with context support
PROMPTS["table_prompt_with_context"] = """
Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented in relation to surrounding context
    - How the table supports or illustrates concepts from the surrounding content
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content.
"""

# Equation analysis prompt template
PROMPTS["equation_prompt"] = """
Please analyze this mathematical equation and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions
    - Mathematical operations and functions used
    - Application domain and context
    - Physical or theoretical significance
    - Relationship to other mathematical concepts
    - Practical applications or use cases
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose and significance (max 100 words)"
    }}
}}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance.
"""

# Equation analysis prompt with context support
PROMPTS["equation_prompt_with_context"] = """
Please analyze this mathematical equation considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions in the context of surrounding content
    - Mathematical operations and functions used
    - Application domain and context based on surrounding material
    - Physical or theoretical significance
    - Relationship to other mathematical concepts mentioned in the context
    - Practical applications or use cases
    - How the equation relates to the broader discussion or framework
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose, significance, and role in the surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance within the broader context.
"""

# Generic content analysis prompt template
PROMPTS["generic_prompt"] = """
Please analyze this {content_type} content and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose and key points (max 100 words)"
    }}
}}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval.
"""

# Generic content analysis prompt with context support
PROMPTS["generic_prompt_with_context"] = """
Please analyze this {content_type} content considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance in relation to surrounding content
    - How this content connects to or supports the broader discussion
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose, key points, and relationship to surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval and understanding the content's role in the broader context.
"""

# Modal chunk templates
PROMPTS["image_chunk"] = """
Image Content Analysis:
- Section Path: {section_path}
- Neighbor Text: {neighbor_text}
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}
"""

PROMPTS["table_chunk"] = """
Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}
"""

PROMPTS["equation_chunk"] = """
Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}
"""

PROMPTS["generic_chunk"] = """
{content_type} Content Analysis:

Content: {content}

Analysis: {enhanced_caption}"""

# Query-related prompts
PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "Please briefly describe the main content, key elements, and important information in this image."
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "You are a professional image analyst who can accurately describe image content."
)

PROMPTS["QUERY_TABLE_ANALYSIS"] = """
Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "You are a professional data analyst who can accurately analyze table data."
)

PROMPTS["QUERY_EQUATION_ANALYSIS"] = """
Please explain the meaning and purpose of the following mathematical formula:

LaTeX formula: {latex}
Formula caption: {equation_caption}

Please briefly explain the mathematical meaning, application scenarios, and importance of this formula."""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "You are a mathematics expert who can clearly explain mathematical formulas."
)

PROMPTS["QUERY_GENERIC_ANALYSIS"] = """
Please analyze the following {content_type} type content and extract its main information and key features:

Content: {content_str}

Please briefly summarize the main characteristics and important information of this content."""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "You are a professional content analyst who can accurately analyze {content_type} type content."
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information."
)