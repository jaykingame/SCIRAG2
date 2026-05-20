import os
import io
import json
import math
from PIL import Image
import fitz  # PyMuPDF: pip install pymupdf
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# OPTION C: Region-Aware Multimodal RAG Pipeline
# ---------------------------------------------------------------------------
# Prerequisites:
# pip install pymupdf pillow google-genai layoutparser torchvision
# 
# Note: For production layout detection, we recommend LayoutParser with Detectron2
# import layoutparser as lp
# ---------------------------------------------------------------------------

class SciRagPipeline:
    def __init__(self, gemini_api_key: str):
        """Initialize the pipeline with Gemini and Layout models."""
        # Initialize the new standard Gemini SDK
        self.client = genai.Client(api_key=gemini_api_key)
        
        # Initialize Layout Model (using LayoutParser + PubLayNet as the standard)
        print("Loading Layout Model (PubLayNet)...")
        # self.layout_model = lp.Detectron2LayoutModel(
        #     config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x',
        #     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        #     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
        # )

    def process_document(self, pdf_path: str):
        """Main pipeline execution for a full PDF document."""
        print(f"Processing document: {pdf_path}")
        doc = fitz.open(pdf_path)
        document_results = []

        for page_num in range(len(doc)):
            print(f"--- Processing Page {page_num + 1} ---")
            page = doc.load_page(page_num)
            
            # Convert PDF page to PIL Image
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # Step 1: Layout Detection
            layout_blocks = self._detect_layout(img, page)
            
            # Step 2: Spatial Heuristics & Linking
            linked_elements = self._link_captions_to_visuals(layout_blocks)
            
            # Step 3: VLM Synthesis (Gemini 1.5 Pro / Gemini 2.5 Flash)
            page_synthesis = self._synthesize_with_vlm(page_num + 1, linked_elements, img)
            
            document_results.append(page_synthesis)

        return document_results

    def _detect_layout(self, img: Image.Image, fitz_page):
        """
        Runs LayoutLM/Detectron2 to find bounding boxes.
        Extracts text for text blocks using PyMuPDF for high accuracy.
        """
        # In a real environment, you run: layout = self.layout_model.detect(img)
        # For this script, we simulate the layout parser output structure.
        
        blocks = []
        
        # Example of how you would process actual layoutparser output:
        # for block in layout:
        #     bbox = block.coordinates # [x1, y1, x2, y2]
        #     block_type = block.type
        #     
        #     text_content = ""
        #     if block_type in ["Text", "Title", "Caption"]:
        #         # Use PyMuPDF to extract exact text within this bounding box
        #         rect = fitz.Rect(bbox)
        #         text_content = fitz_page.get_textbox(rect)
        #
        #     blocks.append({
        #         "type": block_type,
        #         "bbox": bbox,
        #         "text": text_content,
        #         "image_crop": img.crop(bbox) if block_type in ["Figure", "Table"] else None
        #     })
        
        return blocks

    def _link_captions_to_visuals(self, blocks: list):
        """
        Spatial Heuristic Engine:
        Links 'Caption' blocks to the nearest 'Figure' or 'Table' block.
        """
        figures_and_tables = [b for b in blocks if b['type'] in ['Figure', 'Table']]
        captions = [b for b in blocks if b['type'] == 'Caption']
        text_blocks = [b for b in blocks if b['type'] in ['Text', 'Title', 'List']]

        linked_elements = []

        for visual in figures_and_tables:
            v_x1, v_y1, v_x2, v_y2 = visual['bbox']
            best_caption = None
            min_dist = float('inf')

            for cap in captions:
                c_x1, c_y1, c_x2, c_y2 = cap['bbox']
                
                # Heuristic: Caption is usually directly below or above the figure
                # Calculate vertical distance
                dist_below = abs(c_y1 - v_y2)
                dist_above = abs(v_y1 - c_y2)
                
                # Check horizontal overlap
                horizontal_overlap = max(0, min(v_x2, c_x2) - max(v_x1, c_x1))
                
                if horizontal_overlap > 0:
                    dist = min(dist_below, dist_above)
                    if dist < min_dist and dist < 150: # Threshold in pixels
                        min_dist = dist
                        best_caption = cap

            linked_elements.append({
                "visual_type": visual['type'],
                "image": visual['image_crop'],
                "caption_text": best_caption['text'] if best_caption else "No caption found.",
                "bbox": visual['bbox']
            })

        return {
            "linked_visuals": linked_elements,
            "text_context": "\n\n".join([t['text'] for t in text_blocks])
        }

    def _synthesize_with_vlm(self, page_num: int, linked_elements: dict, full_page_img: Image.Image):
        """
        Passes the structured regions to Gemini to generate the grounded JSON.
        """
        prompt = f"""
        You are a scientific multimodal reasoning system.
        You are given structured inputs extracted from page {page_num} of a research paper.

        YOUR TASK:
        Reconstruct the scientific meaning of this page with precise alignment.

        INSTRUCTIONS:
        1. For EACH figure/table provided, describe what is visually shown and explain its scientific purpose.
        2. Use the provided text context to extract claims, methods, and results.
        3. Maintain strict grounding. Do not hallucinate.

        TEXT CONTEXT FROM PAGE:
        {linked_elements['text_context']}
        """

        # Prepare multimodal contents
        contents = [prompt]
        
        for idx, item in enumerate(linked_elements['linked_visuals']):
            contents.append(f"--- {item['visual_type']} {idx + 1} ---")
            contents.append(f"Detected Caption: {item['caption_text']}")
            contents.append(item['image']) # Pass PIL Image directly to Gemini SDK

        # If no visuals were found, just pass the full page image as fallback
        if not linked_elements['linked_visuals']:
            contents.append("Full Page Image (No specific figures detected):")
            contents.append(full_page_img)

        print("Calling Gemini API...")
        
        # We use gemini-2.5-pro for complex reasoning tasks like this
        response = self.client.models.generate_content(
            model='gemini-2.5-pro',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("Failed to parse JSON from VLM.")
            return {"raw_text": response.text}

# Example Usage:
if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("Please set the GEMINI_API_KEY environment variable.")
        exit(1)
        
    pipeline = SciRagPipeline(gemini_api_key=API_KEY)
    
    # results = pipeline.process_document("sample_paper.pdf")
    # print(json.dumps(results, indent=2))
    print("Pipeline initialized successfully.")
