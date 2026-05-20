import React, { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'motion/react';
import { 
  FileText, 
  Scan, 
  Crop, 
  BrainCircuit, 
  Database, 
  ArrowRight,
  Image as ImageIcon,
  Table as TableIcon,
  AlignLeft,
  Link,
  Search,
  Code2,
  Download,
  UploadCloud,
  Loader2,
  CheckCircle2,
  BookOpen,
  Layers,
  ArrowDown
} from 'lucide-react';
import { extractPagesAndText } from './services/pdfService';
import { parsePageWithGemini, ParsedPageResult, generateEmbeddings, askDocument, askDocumentStream, expandQuery, rewriteQuery, IndexItem, generateReport } from './services/geminiService';
import { cropImage, cosineSimilarity } from './utils/ragUtils';
import { tokenize, calculateKeywordScore, reciprocalRankFusion } from './utils/searchUtils';

import { auth, db, signIn, logOut } from './firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import { collection, doc, setDoc, getDocs, query, where, deleteDoc, writeBatch } from 'firebase/firestore';

const stripUndefined = (obj: any): any => {
  if (obj === null || typeof obj !== 'object') return obj;
  if (Array.isArray(obj)) return obj.map(stripUndefined);
  const newObj: any = {};
  for (const key in obj) {
    if (obj[key] !== undefined) {
      newObj[key] = stripUndefined(obj[key]);
    }
  }
  return newObj;
};

const chunkArray = (arr: any[], size: number) => {
  const result = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
};

const steps = [
  {
    id: 'overview',
    title: 'Overview: System Architecture',
    icon: Layers,
    description: 'A high-level data flow diagram illustrating the four-phase architecture of the Sci-RAG system.',
  },
  {
    id: 'layout',
    title: '1. Layout Detection',
    icon: Scan,
    description: 'Detectron2 / LayoutLM identifies bounding boxes for text, figures, tables, and captions on the raw PDF page.',
  },
  {
    id: 'extraction',
    title: '2. Extraction & Linking',
    icon: Crop,
    description: 'Crop regions and use spatial heuristics to link captions to their respective figures or tables.',
  },
  {
    id: 'synthesis',
    title: '3. VLM Synthesis',
    icon: BrainCircuit,
    description: 'Feed structured crops and text into Gemini (Prompt C) to generate a highly structured, grounded JSON representation.',
  },
  {
    id: 'indexing',
    title: '4. Graph Indexing',
    icon: Database,
    description: 'Store the output in a Graph-backed RAG to maintain relationships between text, figures, and tables for fast retrieval.',
  }
];

const pythonCode = `import os
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
            "text_context": "\\n\\n".join([t['text'] for t in text_blocks])
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
    print("Pipeline initialized successfully.")`;

export type ChatMessage = {role: 'user' | 'assistant', text: string, sources?: IndexItem[], status?: string, isStreaming?: boolean};

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [viewMode, setViewMode] = useState<'visualizer' | 'code' | 'parser' | 'chat' | 'report'>('parser');
  const [index, setIndex] = useState<IndexItem[]>([]);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexProgress, setIndexProgress] = useState({ current: 0, total: 0 });
  const [indexError, setIndexError] = useState<string | null>(null);
  const [parsedPages, setParsedPages] = useState<{ image: string, result: ParsedPageResult }[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRestoring, setIsRestoring] = useState(true);
  const [modelIndex, setModelIndex] = useState(0);

  const savedPagesCount = useRef(0);
  const savedIndexCount = useRef(0);
  const savedMessagesCount = useRef(0);

  React.useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setIsAuthReady(true);
    });
    return () => unsubscribe();
  }, []);

  React.useEffect(() => {
    async function loadData() {
      if (!user) {
        setParsedPages([]);
        setIndex([]);
        setMessages([]);
        setIsRestoring(false);
        return;
      }
      try {
        const pagesSnapshot = await getDocs(query(collection(db, 'parsed_pages'), where('userId', '==', user.uid)));
        const storedPages = pagesSnapshot.docs.map(d => d.data() as any);
        
        const indexSnapshot = await getDocs(query(collection(db, 'document_index'), where('userId', '==', user.uid)));
        const storedIndex = indexSnapshot.docs.map(d => d.data() as any);
        
        const messagesSnapshot = await getDocs(query(collection(db, 'chat_messages'), where('userId', '==', user.uid)));
        const storedMessages = messagesSnapshot.docs.map(d => d.data() as any).sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());
        
        if (storedPages && storedPages.length > 0) {
          setParsedPages(storedPages);
          savedPagesCount.current = storedPages.length;
        }
        if (storedIndex && storedIndex.length > 0) {
          setIndex(storedIndex);
          savedIndexCount.current = storedIndex.length;
        }
        if (storedMessages && storedMessages.length > 0) {
          setMessages(storedMessages);
          savedMessagesCount.current = storedMessages.length;
        }
      } catch (e) {
        console.error("Failed to load from Firestore", e);
      } finally {
        setIsRestoring(false);
      }
    }
    if (isAuthReady) {
      loadData();
    }
  }, [user, isAuthReady]);

  React.useEffect(() => {
    if (!isRestoring && user && parsedPages.length > savedPagesCount.current) {
      const newPages = parsedPages.slice(savedPagesCount.current);
      const startIndex = savedPagesCount.current;
      savedPagesCount.current = parsedPages.length;

      const savePages = async () => {
        try {
          const chunks = chunkArray(newPages, 2);
          for (let c = 0; c < chunks.length; c++) {
            const batch = writeBatch(db);
            chunks[c].forEach((page: any, i: number) => {
              const docRef = doc(db, 'parsed_pages', `${user.uid}_page_${startIndex + c * 2 + i}`);
              const cleanPage = stripUndefined({ ...page, userId: user.uid, createdAt: new Date().toISOString() });
              batch.set(docRef, cleanPage);
            });
            await batch.commit();
          }
        } catch (e) {
          console.error("Failed to save pages to Firestore", e);
        }
      };
      savePages();
    }
  }, [parsedPages, isRestoring, user]);

  React.useEffect(() => {
    if (!isRestoring && user && index.length > savedIndexCount.current) {
      const newIndex = index.slice(savedIndexCount.current);
      const startIndex = savedIndexCount.current;
      savedIndexCount.current = index.length;

      const saveIndex = async () => {
        try {
          const chunks = chunkArray(newIndex, 50);
          for (let c = 0; c < chunks.length; c++) {
            const batch = writeBatch(db);
            chunks[c].forEach((item: any, i: number) => {
              const docRef = doc(db, 'document_index', `${user.uid}_index_${startIndex + c * 50 + i}`);
              const cleanItem = stripUndefined({ ...item, userId: user.uid, createdAt: new Date().toISOString() });
              batch.set(docRef, cleanItem);
            });
            await batch.commit();
          }
        } catch (e) {
          console.error("Failed to save index to Firestore", e);
        }
      };
      saveIndex();
    }
  }, [index, isRestoring, user]);

  React.useEffect(() => {
    // For messages, we might update existing messages (e.g., streaming status).
    // So we should just save the last message if it changed, or all messages if length changed.
    // To be safe and simple, we can just save the last message or any message that changed.
    // Actually, messages are appended. We can just save the new ones.
    // Wait, streaming updates the last message. We should always save the last message.
    if (!isRestoring && user && messages.length > 0) {
      const saveMessages = async () => {
        try {
          // Just save the last message to avoid rewriting everything.
          // Or if length increased, save the new ones.
          // Since messages array is small, we can just save the last 2 messages to be safe.
          const messagesToSave = messages.slice(-2);
          const startIndex = Math.max(0, messages.length - 2);
          
          const batch = writeBatch(db);
          messagesToSave.forEach((msg: any, i: number) => {
            const docRef = doc(db, 'chat_messages', `${user.uid}_msg_${startIndex + i}`);
            
            // Strip base64 images from sources to prevent exceeding the 1MB Firestore document limit
            const msgToSave = {
              ...msg,
              sources: msg.sources?.map((s: any) => ({ ...s, imageBase64: undefined }))
            };
            
            const cleanMsg = stripUndefined({ ...msgToSave, userId: user.uid, createdAt: new Date().toISOString() });
            batch.set(docRef, cleanMsg);
          });
          await batch.commit();
        } catch (e) {
          console.error("Failed to save messages to Firestore", e);
        }
      };
      saveMessages();
    }
  }, [messages, isRestoring, user]);

  const handleClearData = async () => {
    if (!user) return;
    if (window.confirm("Are you sure you want to clear all parsed data and chat history?")) {
      try {
        const pagesSnapshot = await getDocs(query(collection(db, 'parsed_pages'), where('userId', '==', user.uid)));
        const indexSnapshot = await getDocs(query(collection(db, 'document_index'), where('userId', '==', user.uid)));
        const messagesSnapshot = await getDocs(query(collection(db, 'chat_messages'), where('userId', '==', user.uid)));
        const reportsSnapshot = await getDocs(query(collection(db, 'reports'), where('userId', '==', user.uid)));
        
        const allDocs = [
          ...pagesSnapshot.docs,
          ...indexSnapshot.docs,
          ...messagesSnapshot.docs,
          ...reportsSnapshot.docs
        ];
        
        const chunks = chunkArray(allDocs, 400);
        for (const chunk of chunks) {
          const batch = writeBatch(db);
          chunk.forEach(d => batch.delete(d.ref));
          await batch.commit();
        }
        
        setParsedPages([]);
        setIndex([]);
        setMessages([]);
        savedPagesCount.current = 0;
        savedIndexCount.current = 0;
        savedMessagesCount.current = 0;
        setViewMode('parser');
      } catch (e) {
        console.error("Failed to clear data", e);
      }
    }
  };

  const buildIndex = async () => {
    if (parsedPages.length === 0 || index.length > 0) return;
    setIsIndexing(true);
    setIndexError(null);
    try {
      const itemsToIndex: Omit<IndexItem, 'embedding'>[] = [];
      for (const page of parsedPages) {
        // Text chunks
        const paragraphs = page.result.key_claims || [];
        for (const p of paragraphs) {
          itemsToIndex.push({ id: Math.random().toString(36).substring(7), type: 'text', pageNumber: page.result.page_number, text: p });
        }
        // Figures
        for (const fig of page.result.figures) {
          let cropped = undefined;
          if (fig.bounding_box_2d) {
            try { cropped = await cropImage(page.image, fig.bounding_box_2d); } catch (e) { console.error(e); }
          }
          itemsToIndex.push({
            id: Math.random().toString(36).substring(7), type: 'figure', pageNumber: page.result.page_number,
            text: `ID: ${fig.id}\nCaption: ${fig.caption}\nDescription: ${fig.visual_description}`,
            imageBase64: cropped,
            boundingBox: fig.bounding_box_2d
          });
        }
        // Tables
        for (const tbl of page.result.tables) {
          let cropped = undefined;
          if (tbl.bounding_box_2d) {
            try { cropped = await cropImage(page.image, tbl.bounding_box_2d); } catch (e) { console.error(e); }
          }
          itemsToIndex.push({
            id: Math.random().toString(36).substring(7), type: 'table', pageNumber: page.result.page_number,
            text: `ID: ${tbl.id}\nCaption: ${tbl.caption}\nDescription: ${tbl.key_findings}`,
            imageBase64: cropped,
            boundingBox: tbl.bounding_box_2d
          });
        }
      }

      setIndexProgress({ current: 0, total: itemsToIndex.length });
      const finalIndex: IndexItem[] = [];
      const batchSize = 5;
      
      for (let i = 0; i < itemsToIndex.length; i += batchSize) {
        const batch = itemsToIndex.slice(i, i + batchSize);
        const contents = batch.map(item => {
          const parts: any[] = [];
          if (item.text) {
            parts.push({ text: item.text });
          }
          if (item.imageBase64) {
            parts.push({ inlineData: { mimeType: 'image/jpeg', data: item.imageBase64 } });
          }
          return { parts };
        });
        
        let success = false;
        let retries = 0;
        while (!success && retries < 3) {
          try {
            const embeddings = await generateEmbeddings(contents);
            batch.forEach((item, idx) => {
              if (embeddings && embeddings[idx] && embeddings[idx].values) {
                finalIndex.push({ ...item, embedding: embeddings[idx].values });
              }
            });
            success = true;
            await new Promise(r => setTimeout(r, 2000));
          } catch (err: any) {
            retries++;
            if (err.message?.includes('429') || err.message?.includes('quota') || err.message?.includes('RESOURCE_EXHAUSTED')) {
              if (retries < 3) {
                await new Promise(r => setTimeout(r, 60000)); // wait 60s
              } else {
                throw new Error("API Quota Exceeded. Please check your plan and billing details at https://ai.google.dev/gemini-api/docs/rate-limits.");
              }
            } else {
              console.error("Embedding error", err);
              throw err;
            }
          }
        }
        setIndexProgress({ current: Math.min(i + batchSize, itemsToIndex.length), total: itemsToIndex.length });
      }
      
      setIndex(finalIndex);
    } catch (err: any) {
      console.error("Failed to build index", err);
      setIndexError(err.message || "An error occurred while building the index.");
    } finally {
      setIsIndexing(false);
    }
  };

  const handleDownloadCode = () => {
    const blob = new Blob([pythonCode], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sci_rag_pipeline.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportChat = () => {
    const chatText = messages.map(m => {
      let text = `**${m.role === 'user' ? 'User' : 'Assistant'}**:\n${m.text}\n`;
      if (m.sources && m.sources.length > 0) {
        text += `\n*Sources used:*\n`;
        m.sources.forEach(s => {
          text += `- Page ${s.pageNumber} (${s.type})\n`;
        });
      }
      return text;
    }).join('\n---\n\n');

    const blob = new Blob([chatText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sci_rag_chat_export.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!isAuthReady || isRestoring) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
          <p className="text-gray-400 font-medium">{!isAuthReady ? 'Checking authentication...' : 'Restoring your workspace...'}</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6">
        <div className="max-w-md w-full bg-[#141414] border border-white/10 rounded-2xl p-8 text-center space-y-6">
          <div className="w-16 h-16 bg-indigo-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <BrainCircuit className="w-8 h-8 text-indigo-400" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Sci-RAG</h1>
          <p className="text-gray-400">
            Sign in to access your multimodal scientific document parser and Q&A workspace.
          </p>
          <button
            onClick={signIn}
            className="w-full py-3 px-4 bg-white text-black hover:bg-gray-200 rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
          >
            Sign in with Google
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-300 font-sans p-8 selection:bg-indigo-500/30">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <header className="mb-12 border-b border-white/10 pb-8 flex justify-between items-end">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="h-8 w-8 rounded-lg bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
                <BrainCircuit className="w-5 h-5 text-indigo-400" />
              </div>
              <h1 className="text-2xl font-semibold text-white tracking-tight">Sci-RAG Architecture</h1>
            </div>
            <p className="text-gray-400 text-sm">Interactive Pipeline Visualizer: Region-Aware Multimodal Reasoning (Option C)</p>
          </div>
          
          {/* View Toggle */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 pr-4 border-r border-white/10">
              {user.photoURL && <img src={user.photoURL} alt="User" className="w-8 h-8 rounded-full border border-white/10" />}
              <button onClick={logOut} className="text-sm text-gray-400 hover:text-white transition-colors">Sign Out</button>
            </div>
            {messages.length > 0 && (
              <button
                onClick={handleExportChat}
                className="px-3 py-1.5 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/20 transition-colors"
                title="Export chat history as Markdown"
              >
                Export Chat
              </button>
            )}
            {parsedPages.length > 0 && (
              <button
                onClick={handleClearData}
                className="px-3 py-1.5 rounded-md text-xs font-medium bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 transition-colors"
                title="Clear all parsed data and chat history"
              >
                Clear Data
              </button>
            )}
            <div className="flex bg-white/5 p-1 rounded-lg border border-white/10">
              <button
                onClick={() => setViewMode('visualizer')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                  viewMode === 'visualizer' ? 'bg-indigo-500/20 text-indigo-300 shadow-sm' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                <Scan className="w-4 h-4" /> Visualizer
              </button>
            <button
              onClick={() => setViewMode('code')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                viewMode === 'code' ? 'bg-indigo-500/20 text-indigo-300 shadow-sm' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <Code2 className="w-4 h-4" /> Python Pipeline
            </button>
            <button
              onClick={() => setViewMode('parser')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                viewMode === 'parser' ? 'bg-emerald-500/20 text-emerald-300 shadow-sm' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <UploadCloud className="w-4 h-4" /> Live Parser
            </button>
            <button
              onClick={() => setViewMode('chat')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                viewMode === 'chat' ? 'bg-amber-500/20 text-amber-300 shadow-sm' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <BrainCircuit className="w-4 h-4" /> Document Q&A
            </button>
            <button
              onClick={() => setViewMode('report')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                viewMode === 'report' ? 'bg-pink-500/20 text-pink-300 shadow-sm' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <BookOpen className="w-4 h-4" /> Report
            </button>
            </div>
          </div>
        </header>

        {viewMode === 'visualizer' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Sidebar - Steps */}
            <div className="lg:col-span-4 space-y-3">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = activeStep === index;
                const isPast = activeStep > index;
                
                return (
                  <button
                    key={step.id}
                    onClick={() => setActiveStep(index)}
                    className={`w-full text-left p-4 rounded-xl border transition-all duration-200 ${
                      isActive 
                        ? 'bg-white/5 border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.1)]' 
                        : isPast
                          ? 'bg-transparent border-white/10 hover:border-white/20'
                          : 'bg-transparent border-transparent opacity-50 hover:opacity-100'
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <div className={`mt-0.5 p-2 rounded-lg ${isActive ? 'bg-indigo-500/20 text-indigo-400' : 'bg-white/5 text-gray-400'}`}>
                        <Icon className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className={`font-medium mb-1 ${isActive ? 'text-white' : 'text-gray-300'}`}>
                          {step.title}
                        </h3>
                        <p className="text-xs text-gray-500 leading-relaxed">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Right Content - Visualization Area */}
            <div className="lg:col-span-8">
              <div className="bg-[#141414] border border-white/10 rounded-2xl p-8 h-[600px] relative overflow-hidden flex items-center justify-center">
                
                {/* Grid Background */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:24px_24px]" />

                <AnimatePresence mode="wait">
                  {activeStep === 0 && <ArchitectureOverviewView key="step0" />}
                  {activeStep === 1 && <LayoutDetectionView key="step1" />}
                  {activeStep === 2 && <ExtractionLinkingView key="step2" />}
                  {activeStep === 3 && <VLMSynthesisView key="step3" />}
                  {activeStep === 4 && <GraphIndexingView key="step4" />}
                </AnimatePresence>

              </div>

              {/* Navigation Controls */}
              <div className="flex justify-between items-center mt-6">
                <button 
                  onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
                  disabled={activeStep === 0}
                  className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                >
                  Previous
                </button>
                <div className="flex gap-2">
                  {steps.map((_, i) => (
                    <div 
                      key={i} 
                      className={`w-2 h-2 rounded-full transition-colors ${i === activeStep ? 'bg-indigo-500' : 'bg-white/20'}`}
                    />
                  ))}
                </div>
                <button 
                  onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
                  disabled={activeStep === steps.length - 1}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-white/10 hover:bg-white/15 text-white disabled:opacity-30 transition-colors flex items-center gap-2"
                >
                  Next Step <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'code' && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-[#141414] border border-white/10 rounded-2xl overflow-hidden shadow-2xl"
          >
            <div className="bg-white/5 px-6 py-4 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Code2 className="w-5 h-5 text-indigo-400" />
                <h2 className="text-white font-medium">sci_rag_pipeline.py</h2>
                <span className="px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-300 text-[10px] font-mono border border-indigo-500/20">
                  Ready for Production
                </span>
              </div>
              <button 
                onClick={handleDownloadCode}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" /> Download Script
              </button>
            </div>
            <div className="p-6 overflow-x-auto max-h-[700px] overflow-y-auto custom-scrollbar">
              <pre className="text-sm font-mono leading-relaxed text-gray-300">
                <code>
                  {pythonCode.split('\n').map((line, i) => {
                    // Very basic syntax highlighting for presentation
                    let highlightedLine = line;
                    if (line.trim().startsWith('#')) {
                      return <div key={i} className="text-gray-500">{line}</div>;
                    }
                    if (line.includes('def ') || line.includes('class ')) {
                      return <div key={i} className="text-blue-400">{line}</div>;
                    }
                    if (line.includes('import ') || line.includes('from ')) {
                      return <div key={i} className="text-purple-400">{line}</div>;
                    }
                    if (line.includes('"""') || line.includes("'''")) {
                      return <div key={i} className="text-green-400/70">{line}</div>;
                    }
                    return <div key={i}>{line}</div>;
                  })}
                </code>
              </pre>
            </div>
          </motion.div>
        )}

        {viewMode === 'parser' && <LiveParserView parsedPages={parsedPages} setParsedPages={setParsedPages} modelIndex={modelIndex} setModelIndex={setModelIndex} />}
        {viewMode === 'report' && (
          <ReportView parsedPages={parsedPages} modelIndex={modelIndex} setModelIndex={setModelIndex} />
        )}
        {viewMode === 'chat' && <ChatView index={index} isIndexing={isIndexing} buildIndex={buildIndex} indexProgress={indexProgress} parsedPages={parsedPages} indexError={indexError} messages={messages} setMessages={setMessages} modelIndex={modelIndex} setModelIndex={setModelIndex} />}
      </div>
    </div>
  );
}

function ReportView({ parsedPages, modelIndex, setModelIndex }: { parsedPages: any[], modelIndex: number, setModelIndex: React.Dispatch<React.SetStateAction<number>> }) {
  const [report, setReport] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const fetchReport = async () => {
      const user = auth.currentUser;
      if (!user) return;
      try {
        const reportsSnapshot = await getDocs(query(collection(db, 'reports'), where('userId', '==', user.uid)));
        if (!reportsSnapshot.empty) {
          setReport(reportsSnapshot.docs[0].data().content);
        }
      } catch (e) {
        console.error("Failed to fetch report", e);
      }
    };
    fetchReport();
  }, []);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    try {
      let currentIdx = modelIndex;
      let success = false;
      let retryCount = 0;
      let generatedText = "";

      while (!success && retryCount < 3) {
        try {
          generatedText = await generateReport(parsedPages, currentIdx);
          success = true;
        } catch (err: any) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('quota') || msg.includes('404') || msg.includes('NOT_FOUND')) {
            currentIdx++;
            setModelIndex(currentIdx);
            retryCount++;
            await new Promise(res => setTimeout(res, 1000));
            continue;
          }
          throw err;
        }
      }
      setReport(generatedText);
      const user = auth.currentUser;
      if (user) {
        const docRef = doc(db, 'reports', `${user.uid}_report`);
        await setDoc(docRef, { content: generatedText, userId: user.uid, createdAt: new Date().toISOString() });
      }
    } catch (err: any) {
      setError(err.message || "Failed to generate report.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!report) return;
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scientific_report.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (parsedPages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-gray-400">
        <BookOpen className="w-12 h-12 mb-4 opacity-50" />
        <p>Please parse a document in the Live Parser first.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[700px] bg-[#141414] border border-white/10 rounded-2xl overflow-hidden p-6">
      <div className="flex justify-between items-center mb-6 border-b border-white/10 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-white">Document Report</h2>
          <p className="text-sm text-gray-400">Generate a comprehensive summary of the parsed document.</p>
        </div>
        <div className="flex gap-3">
          {report && (
            <button onClick={handleDownload} className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg text-sm font-medium transition-colors">
              Download Markdown
            </button>
          )}
          <button 
            onClick={handleGenerate} 
            disabled={isGenerating}
            className="px-4 py-2 bg-pink-600 hover:bg-pink-700 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <BookOpen className="w-4 h-4" />}
            {isGenerating ? 'Generating...' : report ? 'Regenerate Report' : 'Generate Report'}
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto custom-scrollbar pr-4">
        {error && (
          <div className="p-4 bg-red-500/20 border border-red-500/30 text-red-400 rounded-xl mb-4">
            {error}
          </div>
        )}
        
        {isGenerating ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 space-y-4">
            <Loader2 className="w-8 h-8 animate-spin text-pink-500" />
            <p>Synthesizing extracted data into a comprehensive report...</p>
          </div>
        ) : report ? (
          <div className="markdown-body text-gray-200">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <p>Click "Generate Report" to create a summary of the parsed document.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function ChatView({ index, isIndexing, buildIndex, indexProgress, parsedPages, indexError, messages, setMessages, modelIndex, setModelIndex }: { index: IndexItem[], isIndexing: boolean, buildIndex: () => void, indexProgress: any, parsedPages: any[], indexError: string | null, messages: ChatMessage[], setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>, modelIndex: number, setModelIndex: React.Dispatch<React.SetStateAction<number>> }) {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeSource, setActiveSource] = useState<IndexItem | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  React.useEffect(() => {
    if (parsedPages.length > 0 && index.length === 0 && !isIndexing && !indexError) {
      buildIndex();
    }
  }, [parsedPages, index, isIndexing, indexError]);

  const renderMessageText = (text: string, sources?: IndexItem[]) => {
    if (!sources) {
      return (
        <div className="markdown-body text-sm leading-relaxed text-gray-200">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        </div>
      );
    }
    
    // Pre-process text to convert citations to links
    const processedText = text.replace(/\[Page (\d+), ([A-Za-z]+)\]/g, (match, pageNum, type) => {
      return `[${match}](#cite-${pageNum}-${type.toLowerCase()})`;
    });

    return (
      <div className="markdown-body text-sm leading-relaxed text-gray-200">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({node, href, children, ...props}) => {
              if (href?.startsWith('#cite-')) {
                const [, pageNumStr, type] = href.split('-');
                const pageNum = parseInt(pageNumStr);
                
                const source = sources.find(s => s.pageNumber === pageNum && s.type === type) || 
                               sources.find(s => s.pageNumber === pageNum);
                
                if (source) {
                  return (
                    <button 
                      onClick={() => setActiveSource(source)}
                      className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-1 bg-indigo-500/20 hover:bg-indigo-500/40 text-indigo-300 rounded text-xs font-medium transition-colors border border-indigo-500/30"
                    >
                      <Link className="w-3 h-3" />
                      {children?.toString().replace('[', '').replace(']', '')}
                    </button>
                  );
                }
                return <span className="text-indigo-300">{children}</span>;
              }
              return <a href={href} target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline" {...props}>{children}</a>;
            },
            p: ({children}) => <p className="mb-4 last:mb-0">{children}</p>,
            ul: ({children}) => <ul className="list-disc pl-6 mb-4">{children}</ul>,
            ol: ({children}) => <ol className="list-decimal pl-6 mb-4">{children}</ol>,
            li: ({children}) => <li className="mb-1">{children}</li>,
            h1: ({children}) => <h1 className="text-xl font-bold mb-4 mt-6 text-white">{children}</h1>,
            h2: ({children}) => <h2 className="text-lg font-bold mb-3 mt-5 text-white">{children}</h2>,
            h3: ({children}) => <h3 className="text-md font-bold mb-2 mt-4 text-white">{children}</h3>,
            strong: ({children}) => <strong className="font-bold text-white">{children}</strong>,
            em: ({children}) => <em className="italic">{children}</em>,
            code: ({inline, className, children, ...props}: any) => {
              const match = /language-(\w+)/.exec(className || '');
              return inline ? (
                <code className="bg-white/10 px-1 py-0.5 rounded text-sm font-mono text-indigo-200" {...props}>
                  {children}
                </code>
              ) : (
                <pre className="bg-black/40 p-4 rounded-xl overflow-x-auto mb-4 border border-white/10">
                  <code className="text-sm font-mono text-gray-300" {...props}>
                    {children}
                  </code>
                </pre>
              );
            }
          }}
        >
          {processedText}
        </ReactMarkdown>
      </div>
    );
  };

  const handleSend = async () => {
    if (!input.trim() || isTyping || index.length === 0) return;
    
    const question = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: question }]);
    setIsTyping(true);

    abortControllerRef.current = new AbortController();

    try {
      // Add initial assistant message with status
      setMessages(prev => [...prev, { role: 'assistant', text: '', status: 'Understanding context...', isStreaming: true }]);

      let currentIdx = modelIndex;

      // Helper for retrying with model rotation
      async function withRetry<T>(fn: (idx: number) => Promise<T>, maxRetries = 3): Promise<T> {
        let lastErr: any;
        for (let i = 0; i < maxRetries; i++) {
          try {
            return await fn(currentIdx);
          } catch (err: any) {
            lastErr = err;
            const msg = err instanceof Error ? err.message : String(err);
            if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('quota') || msg.includes('exceeded your current quota') || msg.includes('404') || msg.includes('NOT_FOUND')) {
              console.log("Rate limit or model not found hit in chat, rotating model...");
              currentIdx++;
              setModelIndex(currentIdx);
              await new Promise(res => setTimeout(res, 1000));
              continue;
            }
            throw err;
          }
        }
        throw lastErr;
      }

      // 0. Contextual Query Rewriting
      const chatHistory = messages.map(m => ({ role: m.role, text: m.text }));
      const rewrittenQuery = await withRetry((idx) => rewriteQuery(question, chatHistory, idx));

      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].status = 'Expanding query for better retrieval...';
        return newMsgs;
      });

      // 1. Query Expansion
      const expandedQueries = await withRetry((idx) => expandQuery(rewrittenQuery, idx));
      const allQueries = Array.from(new Set([rewrittenQuery, ...expandedQueries]));

      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].status = 'Embedding queries...';
        return newMsgs;
      });

      // 2. Embed all queries
      const qEmbResults = await generateEmbeddings(allQueries);

      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].status = 'Performing hybrid search...';
        return newMsgs;
      });

      // 3. Hybrid Search (Semantic + Keyword) across all queries
      const rankedLists: IndexItem[][] = [];

      for (let i = 0; i < allQueries.length; i++) {
        const qStr = allQueries[i];
        const qEmb = qEmbResults[i].values;
        const qTokens = tokenize(qStr);

        const scored = index.map(item => {
          let semanticScore = cosineSimilarity(qEmb, item.embedding);
          const itemTokens = tokenize(item.text);
          
          // Boost keyword matching by including the type explicitly
          if (item.type === 'figure') itemTokens.push('figure', 'image', 'diagram', 'chart', 'plot');
          if (item.type === 'table') itemTokens.push('table', 'data');

          let keywordScore = calculateKeywordScore(qTokens, itemTokens);

          // Explicit intent boosting
          const qLower = qStr.toLowerCase();
          if (item.type === 'figure' && (qLower.includes('figure') || qLower.includes('image') || qLower.includes('diagram') || qLower.includes('chart'))) {
            semanticScore += 0.15; // Significant boost to semantic score
            keywordScore += 1.0;   // Significant boost to keyword score
          }
          if (item.type === 'table' && (qLower.includes('table') || qLower.includes('data'))) {
            semanticScore += 0.15;
            keywordScore += 1.0;
          }

          return { item, semanticScore, keywordScore };
        });

        // List 1: Ranked by Semantic
        const semanticRanked = [...scored].sort((a, b) => b.semanticScore - a.semanticScore).map(s => s.item);
        // List 2: Ranked by Keyword
        const keywordRanked = [...scored].sort((a, b) => b.keywordScore - a.keywordScore).map(s => s.item);

        rankedLists.push(semanticRanked);
        rankedLists.push(keywordRanked);
      }

      // 4. Reciprocal Rank Fusion (RRF)
      const fusedResults = reciprocalRankFusion(rankedLists);

      // 5. Retrieve top relevant items across the entire document
      const topItems = fusedResults
        .map(f => f.item)
        .slice(0, 5); // Lowered to 5 for more focused retrieval

      // Find the unique pages these top items belong to
      const uniquePageNums = Array.from(new Set(topItems.map(item => item.pageNumber)));

      const topK: IndexItem[] = [];
      
      // Add the full page images as context items for the retrieved chunks
      uniquePageNums.forEach(pageNum => {
        const pageData = parsedPages.find(p => p.result.page_number === pageNum);
        if (pageData) {
          topK.push({
            id: `page-${pageNum}`,
            type: 'page',
            pageNumber: pageNum,
            text: `Full context of Page ${pageNum}`,
            imageBase64: pageData.image,
            embedding: []
          });
        }
      });

      // Add the specific highly relevant chunks
      topK.push(...topItems);

      // 6. Streaming Response
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].status = undefined; // Clear status
        newMsgs[newMsgs.length - 1].sources = topK;
        return newMsgs;
      });

      // Pass the previous messages as chat history (excluding the current empty assistant message and the user's current question)
      const streamChatHistory = messages.slice(0, -1).map(m => ({ role: m.role, text: m.text }));
      
      let fullText = "";
      let success = false;
      let retryCount = 0;
      const maxRetries = 3;

      while (!success && retryCount < maxRetries) {
        try {
          const stream = askDocumentStream(question, topK, streamChatHistory, abortControllerRef.current?.signal, currentIdx);
          for await (const chunk of stream) {
            if (abortControllerRef.current?.signal.aborted) break;
            fullText += chunk;
            setMessages(prev => {
              const newMsgs = [...prev];
              newMsgs[newMsgs.length - 1].text = fullText;
              return newMsgs;
            });
          }
          success = true;
        } catch (err: any) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('quota') || msg.includes('exceeded your current quota') || msg.includes('404') || msg.includes('NOT_FOUND')) {
            console.log("Rate limit or model not found hit in stream, rotating model...");
            currentIdx++;
            setModelIndex(currentIdx);
            retryCount++;
            fullText = ""; // Reset text for retry
            if (retryCount < maxRetries) {
              await new Promise(res => setTimeout(res, 1000));
              continue;
            }
          }
          throw err;
        }
      }

      // Finalize
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].isStreaming = false;
        return newMsgs;
      });

      // Save to IndexedDB
      // Removed local set call, Firestore useEffect handles saving

    } catch (err: any) {
      if (err.name === 'AbortError' || abortControllerRef.current?.signal.aborted) {
        console.log('Generation cancelled');
        setMessages(prev => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1].isStreaming = false;
          return newMsgs;
        });
      } else {
        console.error(err);
        const errMsg = err.message?.includes('429') || err.message?.includes('quota') 
          ? "API Quota Exceeded. Please check your plan and billing details."
          : "Sorry, I encountered an error while answering.";
        
        setMessages(prev => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1].text = errMsg;
          newMsgs[newMsgs.length - 1].status = undefined;
          newMsgs[newMsgs.length - 1].isStreaming = false;
          return newMsgs;
        });
      }
    } finally {
      setIsTyping(false);
      abortControllerRef.current = null;
    }
  };

  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  if (parsedPages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-gray-400">
        <BrainCircuit className="w-12 h-12 mb-4 opacity-50" />
        <p>Please parse a document in the Live Parser first.</p>
      </div>
    );
  }

  if (indexError) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-red-400">
        <div className="bg-red-500/10 border border-red-500/20 p-6 rounded-2xl max-w-lg text-center">
          <BrainCircuit className="w-12 h-12 mb-4 mx-auto opacity-80" />
          <h3 className="text-xl font-medium mb-2">Indexing Failed</h3>
          <p className="text-sm opacity-80 mb-4">{indexError}</p>
          <button 
            onClick={buildIndex}
            className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-colors text-sm font-medium"
          >
            Retry Indexing
          </button>
        </div>
      </div>
    );
  }

  if (isIndexing) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-gray-400">
        <Loader2 className="w-12 h-12 mb-4 animate-spin text-indigo-500" />
        <h3 className="text-xl text-white mb-2">Building Multimodal Index...</h3>
        <p>Embedding text, figures, and tables into Gemini Embedding 2.</p>
        <p className="text-sm mt-2">Progress: {indexProgress.current} / {indexProgress.total} items</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[700px] bg-[#141414] border border-white/10 rounded-2xl overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
        {messages.length === 0 && (
          <div className="text-center py-10 text-gray-400">
            <BrainCircuit className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <h3 className="text-xl text-white mb-2">Document Q&A Ready</h3>
            <p>Ask questions about the paper. The system will retrieve relevant text, figures, and tables to answer.</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl p-4 ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-200 border border-white/10'}`}>
              {msg.status && (
                <div className="flex items-center gap-2 mb-2 text-indigo-400 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>{msg.status}</span>
                </div>
              )}
              {renderMessageText(msg.text, msg.sources)}
              {msg.isStreaming && !msg.status && (
                <span className="inline-block w-2 h-4 ml-1 bg-indigo-400 animate-pulse" />
              )}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/10">
                  <p className="text-xs text-gray-400 mb-2 uppercase tracking-wider font-semibold">Retrieved Sources</p>
                  <div className="flex flex-wrap gap-2">
                    {msg.sources.map((src, sIdx) => (
                      <button 
                        key={sIdx} 
                        onClick={() => setActiveSource(src)}
                        className="bg-black/40 hover:bg-black/60 rounded p-2 text-xs border border-white/5 max-w-[200px] text-left transition-colors"
                      >
                        <div className="flex items-center gap-1 mb-1 text-indigo-300">
                          {src.type === 'figure' ? <ImageIcon className="w-3 h-3" /> : src.type === 'table' ? <TableIcon className="w-3 h-3" /> : src.type === 'page' ? <FileText className="w-3 h-3" /> : <AlignLeft className="w-3 h-3" />}
                          <span className="capitalize">Page {src.pageNumber} {src.type}</span>
                        </div>
                        {src.imageBase64 && (
                          <img src={`data:image/jpeg;base64,${src.imageBase64}`} alt="Source" className="w-full h-auto rounded mb-1" />
                        )}
                        <p className="line-clamp-3 text-gray-400">{src.text}</p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        {isTyping && messages[messages.length - 1]?.role === 'user' && (
          <div className="flex justify-start">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-4 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
              <span className="text-gray-400 text-sm">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-4 border-t border-white/10 bg-black/20">
        <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="flex gap-2">
          <input 
            type="text" 
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about the paper..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
          />
          {isTyping ? (
            <button 
              type="button"
              onClick={handleStopGeneration}
              className="bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 px-6 py-3 rounded-xl font-medium transition-colors flex items-center gap-2"
            >
              Stop
            </button>
          ) : (
            <button 
              type="submit"
              disabled={!input.trim()}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white px-6 py-3 rounded-xl font-medium transition-colors"
            >
              Send
            </button>
          )}
        </form>
      </div>

      <AnimatePresence>
        {activeSource && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8"
            onClick={() => setActiveSource(null)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-[#1a1a1a] border border-white/10 rounded-2xl overflow-hidden shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-4 border-b border-white/10 flex justify-between items-center bg-black/20">
                <div className="flex items-center gap-2 text-indigo-300">
                  {activeSource.type === 'figure' ? <ImageIcon className="w-5 h-5" /> : activeSource.type === 'table' ? <TableIcon className="w-5 h-5" /> : activeSource.type === 'page' ? <FileText className="w-5 h-5" /> : <AlignLeft className="w-5 h-5" />}
                  <h3 className="font-medium capitalize">Page {activeSource.pageNumber} {activeSource.type}</h3>
                </div>
                <button onClick={() => setActiveSource(null)} className="text-gray-400 hover:text-white">
                  Close
                </button>
              </div>
              <div className="p-6 overflow-y-auto flex-1 custom-scrollbar flex flex-col items-center">
                {activeSource.type !== 'page' && activeSource.boundingBox && (
                  <div className="w-full mb-6 relative">
                    <p className="text-sm text-gray-400 mb-2">Context on Full Page:</p>
                    <div className="relative inline-block border border-white/10 rounded overflow-hidden">
                      <img 
                        src={`data:image/jpeg;base64,${parsedPages.find(p => p.result.page_number === activeSource.pageNumber)?.image}`} 
                        alt="Full Page" 
                        className="max-w-full max-h-[60vh] object-contain opacity-50"
                      />
                      <div 
                        className="absolute border-2 border-indigo-500 bg-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.5)]"
                        style={{
                          top: `${activeSource.boundingBox[0] / 10}%`,
                          left: `${activeSource.boundingBox[1] / 10}%`,
                          height: `${(activeSource.boundingBox[2] - activeSource.boundingBox[0]) / 10}%`,
                          width: `${(activeSource.boundingBox[3] - activeSource.boundingBox[1]) / 10}%`
                        }}
                      />
                    </div>
                  </div>
                )}
                
                {activeSource.imageBase64 && activeSource.type !== 'page' && (
                  <div className="w-full mb-6">
                    <p className="text-sm text-gray-400 mb-2">Cropped Region:</p>
                    <img src={`data:image/jpeg;base64,${activeSource.imageBase64}`} alt="Source Crop" className="max-w-full rounded border border-white/10" />
                  </div>
                )}
                
                {activeSource.type === 'page' && activeSource.imageBase64 && (
                  <div className="w-full mb-6">
                    <img src={`data:image/jpeg;base64,${activeSource.imageBase64}`} alt="Full Page" className="max-w-full rounded border border-white/10" />
                  </div>
                )}

                <div className="w-full bg-black/40 rounded-xl p-4 border border-white/5">
                  <p className="text-sm text-gray-300 whitespace-pre-wrap">{activeSource.text}</p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function CroppedImage({ base64Image, boundingBox }: { base64Image: string, boundingBox?: number[] }) {
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
  
  React.useEffect(() => {
    const img = new Image();
    img.onload = () => setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = `data:image/jpeg;base64,${base64Image}`;
  }, [base64Image]);

  if (!boundingBox || boundingBox.length !== 4) return null;
  if (!imgDims.w) return <div className="animate-pulse bg-white/5 h-48 rounded-lg w-full my-4" />;

  const [ymin, xmin, ymax, xmax] = boundingBox;
  
  const top = (ymin / 1000) * 100;
  const left = (xmin / 1000) * 100;
  const bottom = (ymax / 1000) * 100;
  const right = (xmax / 1000) * 100;
  
  const width = Math.max(right - left, 1);
  const height = Math.max(bottom - top, 1);

  const cropW = (width / 100) * imgDims.w;
  const cropH = (height / 100) * imgDims.h;
  
  return (
    <div 
      className="relative overflow-hidden rounded-lg border border-white/10 bg-black/50 my-4"
      style={{ 
        width: '100%', 
        paddingBottom: `${(cropH / cropW) * 100}%` 
      }}
    >
      <img 
        src={`data:image/jpeg;base64,${base64Image}`} 
        alt="Cropped"
        className="absolute max-w-none"
        style={{
          width: `${(100 / width) * 100}%`,
          height: `${(100 / height) * 100}%`,
          top: `-${(top / height) * 100}%`,
          left: `-${(left / width) * 100}%`,
        }}
      />
    </div>
  );
}

// --- Live Parser Component ---
function LiveParserView({ parsedPages, setParsedPages, modelIndex, setModelIndex }: { parsedPages: any[], setParsedPages: any, modelIndex: number, setModelIndex: React.Dispatch<React.SetStateAction<number>> }) {
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<'idle' | 'converting' | 'parsing' | 'waiting' | 'done' | 'error'>(
    parsedPages.length > 0 ? 'done' : 'idle'
  );
  const [errorMsg, setErrorMsg] = useState('');
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (parsedPages.length === 0 && status === 'done') {
      setStatus('idle');
    }
  }, [parsedPages, status]);

  const handleFile = async (file: File) => {
    if (file.type !== 'application/pdf') {
      setErrorMsg('Please upload a valid PDF file.');
      setStatus('error');
      return;
    }

    try {
      setStatus('converting');
      setErrorMsg('');
      setParsedPages([]);
      
      // 1. Convert PDF to Images (limited to 50 pages during rasterization to prevent memory crash)
      const { images, texts, totalPages } = await extractPagesAndText(file, 50);
      
      if (totalPages > 50) {
        setErrorMsg(`Warning: This document has ${totalPages} pages. To prevent browser crashes and API quota exhaustion, only the first 50 pages will be processed.`);
      }
      
      setProgress({ current: 0, total: images.length });
      setStatus('parsing');

      // 2. Parse each page sequentially with Gemini
      const results: { image: string, result: ParsedPageResult }[] = [];
      let currentIdx = modelIndex;
      
      for (let i = 0; i < images.length; i++) {
        setProgress({ current: i + 1, total: images.length });
        
        let success = false;
        let retryCount = 0;
        const maxRetries = 3;

        while (!success && retryCount < maxRetries) {
          try {
            setStatus('parsing');
            const result = await parsePageWithGemini(images[i], i + 1, texts[i], currentIdx);
            const newPage = { image: images[i], result };
            results.push(newPage);
            
            // Update state progressively so user sees results coming in
            setParsedPages(prev => [...prev, newPage]);
            success = true;
          } catch (err: any) {
            console.error(`Error parsing page ${i + 1}:`, err);
            const errorMessage = err instanceof Error ? err.message : String(err);
            
            // Check for rate limit, quota, or not found error
            if (errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED') || errorMessage.includes('quota') || errorMessage.includes('exceeded your current quota') || errorMessage.includes('404') || errorMessage.includes('NOT_FOUND')) {
              // Rotate model and retry
              console.log("Rate limit or model not found hit, rotating model...");
              currentIdx++;
              setModelIndex(currentIdx);
              retryCount++;
              
              if (retryCount < maxRetries) {
                setStatus('waiting');
                // Wait a bit before retrying with the new model
                await new Promise(res => setTimeout(res, 2000));
                continue;
              } else {
                throw new Error("All models reached their limits or were not found. Please check your plan and billing details at https://ai.google.dev/gemini-api/docs/rate-limits.");
              }
            } else {
              // Not a rate limit error, throw to stop
              throw err;
            }
          }
        }
        
        // Add a delay between successful requests to avoid hitting limits too fast
        if (success && i < images.length - 1) {
          await new Promise(res => setTimeout(res, 2000));
        }
      }
      
      setStatus('done');
    } catch (err) {
      console.error(err);
      setErrorMsg(err instanceof Error ? err.message : 'An error occurred during processing.');
      setStatus('error');
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      {status === 'idle' || status === 'error' ? (
        <div 
          className={`border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center transition-colors cursor-pointer ${
            isDragging ? 'border-emerald-500 bg-emerald-500/10' : 'border-white/20 bg-white/5 hover:bg-white/10'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
              handleFile(e.dataTransfer.files[0]);
            }
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept="application/pdf"
            onChange={(e) => e.target.files && handleFile(e.target.files[0])}
          />
          <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center mb-4">
            <UploadCloud className="w-8 h-8 text-emerald-400" />
          </div>
          <h3 className="text-xl font-medium text-white mb-2">Upload Scientific PDF</h3>
          <p className="text-gray-400 text-center max-w-md">
            Drag and drop your research paper here. Gemini 3.1 Flash Lite will parse it page-by-page, extracting figures, tables, and scientific insights.
          </p>
          {status === 'error' && (
            <div className="mt-4 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg text-sm border border-red-500/30">
              {errorMsg}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-[#141414] border border-white/10 rounded-2xl p-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {status === 'done' ? (
              <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
            ) : status === 'waiting' ? (
              <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
                <Loader2 className="w-5 h-5 text-amber-400 animate-spin" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center">
                <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
              </div>
            )}
            <div>
              <h3 className="text-white font-medium">
                {status === 'converting' && 'Rasterizing PDF Pages...'}
                {status === 'parsing' && `Parsing Page ${progress.current} of ${progress.total}...`}
                {status === 'waiting' && `Rate limit reached. Pausing for 60s before Page ${progress.current}...`}
                {status === 'done' && 'Processing Complete'}
              </h3>
              <p className="text-sm text-gray-400">
                {status === 'parsing' && 'Gemini 3.1 Flash Lite is extracting figures, tables, and insights.'}
                {status === 'waiting' && 'The API quota was exceeded. Automatically retrying shortly.'}
                {status === 'done' && `Successfully parsed ${parsedPages.length} pages.`}
              </p>
            </div>
          </div>
          {status === 'done' && (
            <button 
              onClick={() => setStatus('idle')}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Parse Another Document
            </button>
          )}
        </div>
      )}

      {/* Results View */}
      {parsedPages.length > 0 && (
        <div className="space-y-8 mt-8">
          {parsedPages.map((page, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-[#141414] border border-white/10 rounded-2xl overflow-hidden flex flex-col lg:flex-row"
            >
              {/* Left: Page Image */}
              <div className="lg:w-1/3 bg-black/50 p-6 flex items-center justify-center border-r border-white/10">
                <img 
                  src={`data:image/jpeg;base64,${page.image}`} 
                  alt={`Page ${page.result.page_number}`}
                  className="max-w-full h-auto rounded shadow-2xl border border-white/5"
                />
              </div>
              
              {/* Right: Extracted Data */}
              <div className="lg:w-2/3 p-6 space-y-6 overflow-y-auto max-h-[800px] custom-scrollbar">
                <div className="flex items-center justify-between border-b border-white/10 pb-4">
                  <h3 className="text-xl font-semibold text-white">Page {page.result.page_number}</h3>
                  <span className="px-3 py-1 bg-indigo-500/20 text-indigo-300 text-xs font-mono rounded-full border border-indigo-500/30">
                    Gemini 3.1 Flash Lite
                  </span>
                </div>

                {/* Figures */}
                {page.result.figures && page.result.figures.length > 0 && (
                  <div className="space-y-4">
                    <h4 className="text-sm font-mono text-violet-400 flex items-center gap-2 uppercase tracking-wider">
                      <ImageIcon className="w-4 h-4" /> Extracted Figures
                    </h4>
                    {page.result.figures.map((fig, i) => (
                      <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-white">{fig.id}</span>
                        </div>
                        <CroppedImage base64Image={page.image} boundingBox={fig.bounding_box_2d} />
                        <p className="text-sm text-gray-300"><span className="text-gray-500">Caption:</span> {fig.caption}</p>
                        <p className="text-sm text-gray-300"><span className="text-gray-500">Visual:</span> {fig.visual_description}</p>
                        <div className="bg-violet-500/10 border border-violet-500/20 rounded-lg p-3">
                          <p className="text-sm text-violet-200"><span className="font-semibold text-violet-400">Scientific Insight:</span> {fig.scientific_insight}</p>
                        </div>
                        {fig.linked_text_quotes && fig.linked_text_quotes.length > 0 && (
                          <div className="text-xs text-gray-500 italic border-l-2 border-white/10 pl-3">
                            "{fig.linked_text_quotes[0]}"
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Tables */}
                {page.result.tables && page.result.tables.length > 0 && (
                  <div className="space-y-4">
                    <h4 className="text-sm font-mono text-amber-400 flex items-center gap-2 uppercase tracking-wider">
                      <TableIcon className="w-4 h-4" /> Extracted Tables
                    </h4>
                    {page.result.tables.map((table, i) => (
                      <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-white">{table.id}</span>
                        </div>
                        <CroppedImage base64Image={page.image} boundingBox={table.bounding_box_2d} />
                        <p className="text-sm text-gray-300"><span className="text-gray-500">Caption:</span> {table.caption}</p>
                        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
                          <p className="text-sm text-amber-200"><span className="font-semibold text-amber-400">Key Findings:</span> {table.key_findings}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Key Claims */}
                {page.result.key_claims && page.result.key_claims.length > 0 && (
                  <div className="space-y-4">
                    <h4 className="text-sm font-mono text-emerald-400 flex items-center gap-2 uppercase tracking-wider">
                      <FileText className="w-4 h-4" /> Key Scientific Claims
                    </h4>
                    <ul className="list-disc list-inside space-y-2 text-sm text-gray-300">
                      {page.result.key_claims.map((claim, i) => (
                        <li key={i}>{claim}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Empty State */}
                {(!page.result.figures || page.result.figures.length === 0) && 
                 (!page.result.tables || page.result.tables.length === 0) && 
                 (!page.result.key_claims || page.result.key_claims.length === 0) && (
                  <div className="text-center py-8 text-gray-500 text-sm">
                    No figures, tables, or major claims detected on this page.
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Visualization Components ---

function ArchitectureOverviewView() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="w-full max-w-2xl relative z-10 flex flex-col items-center"
    >
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">Figure 1: Sci-RAG System Architecture</h3>
        <p className="text-gray-400 text-sm max-w-md mx-auto">
          A high-level data flow diagram illustrating the four-phase architecture. The pipeline begins with PDF rasterization, flows through the Gemini 3.1 Flash Lite multimodal parsing engine, enters the dual-indexing phase (text and visual vectors), and culminates in the Advanced Retrieval and Grounded Generation modules.
        </p>
      </div>

      <div className="flex flex-col items-center w-full space-y-4">
        {/* Phase 1 */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4 w-full flex items-center gap-4">
          <div className="bg-blue-500/20 p-3 rounded-lg text-blue-400">
            <FileText className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">1. PDF Rasterization</h4>
            <p className="text-xs text-gray-400">Convert raw PDF documents into high-resolution images.</p>
          </div>
        </div>

        <ArrowDown className="w-5 h-5 text-gray-500" />

        {/* Phase 2 */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4 w-full flex items-center gap-4">
          <div className="bg-purple-500/20 p-3 rounded-lg text-purple-400">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">2. Gemini 3.1 Flash Lite</h4>
            <p className="text-xs text-gray-400">Multimodal parsing engine extracts text, figures, and tables.</p>
          </div>
        </div>

        <ArrowDown className="w-5 h-5 text-gray-500" />

        {/* Phase 3 */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4 w-full flex items-center gap-4">
          <div className="bg-green-500/20 p-3 rounded-lg text-green-400">
            <Database className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">3. Dual-Indexing Phase</h4>
            <p className="text-xs text-gray-400">Generate and store text and visual vectors for semantic search.</p>
          </div>
        </div>

        <ArrowDown className="w-5 h-5 text-gray-500" />

        {/* Phase 4 */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4 w-full flex items-center gap-4">
          <div className="bg-orange-500/20 p-3 rounded-lg text-orange-400">
            <Search className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">4. Advanced Retrieval & Grounded Generation</h4>
            <p className="text-xs text-gray-400">Retrieve relevant context and generate accurate, cited responses.</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function LayoutDetectionView() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="relative z-10 w-full max-w-md"
    >
      <div className="bg-white p-6 rounded-sm shadow-2xl aspect-[1/1.4] relative">
        {/* Mock PDF Content */}
        <div className="space-y-4">
          {/* Title */}
          <motion.div 
            initial={{ borderColor: 'transparent' }}
            animate={{ borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
            transition={{ delay: 0.5 }}
            className="border-2 border-dashed p-2 rounded-sm relative group"
          >
            <div className="h-4 bg-gray-300 rounded w-3/4 mb-2" />
            <div className="h-4 bg-gray-300 rounded w-1/2" />
            <span className="absolute -top-3 -left-2 bg-blue-500 text-white text-[10px] px-1.5 py-0.5 rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">Title</span>
          </motion.div>

          {/* Text Block */}
          <motion.div 
            initial={{ borderColor: 'transparent' }}
            animate={{ borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)' }}
            transition={{ delay: 0.8 }}
            className="border-2 border-dashed p-2 rounded-sm relative group"
          >
            <div className="space-y-2">
              <div className="h-2 bg-gray-200 rounded w-full" />
              <div className="h-2 bg-gray-200 rounded w-full" />
              <div className="h-2 bg-gray-200 rounded w-5/6" />
            </div>
            <span className="absolute -top-3 -left-2 bg-emerald-500 text-white text-[10px] px-1.5 py-0.5 rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">Text</span>
          </motion.div>

          {/* Figure */}
          <motion.div 
            initial={{ borderColor: 'transparent' }}
            animate={{ borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)' }}
            transition={{ delay: 1.1 }}
            className="border-2 border-dashed p-2 rounded-sm relative group"
          >
            <div className="h-32 bg-gray-100 border border-gray-200 rounded flex items-center justify-center">
              <ImageIcon className="w-8 h-8 text-gray-400" />
            </div>
            <span className="absolute -top-3 -left-2 bg-violet-500 text-white text-[10px] px-1.5 py-0.5 rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">Figure</span>
          </motion.div>

          {/* Caption */}
          <motion.div 
            initial={{ borderColor: 'transparent' }}
            animate={{ borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)' }}
            transition={{ delay: 1.4 }}
            className="border-2 border-dashed p-1.5 rounded-sm relative group"
          >
            <div className="h-2 bg-gray-300 rounded w-full mb-1" />
            <div className="h-2 bg-gray-300 rounded w-2/3" />
            <span className="absolute -top-3 -left-2 bg-amber-500 text-white text-[10px] px-1.5 py-0.5 rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">Caption</span>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
}

function ExtractionLinkingView() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="relative z-10 w-full max-w-2xl flex items-center justify-between gap-8"
    >
      {/* Extracted Elements */}
      <div className="flex-1 space-y-6">
        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-[#1a1a1a] border border-violet-500/30 p-4 rounded-xl relative"
        >
          <div className="flex items-center gap-2 mb-3 text-violet-400 text-xs font-mono uppercase">
            <ImageIcon className="w-4 h-4" /> Cropped Figure
          </div>
          <div className="h-24 bg-white/5 rounded border border-white/10 flex items-center justify-center">
            <div className="w-16 h-16 rounded-full border-4 border-violet-500/50 border-t-transparent animate-spin" />
          </div>
          
          {/* Linking Line */}
          <svg className="absolute -right-12 top-1/2 w-12 h-24 overflow-visible" style={{ transform: 'translateY(-50%)' }}>
            <motion.path 
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ delay: 1, duration: 0.5 }}
              d="M 0 0 C 20 0, 20 96, 48 96" 
              fill="none" 
              stroke="#f59e0b" 
              strokeWidth="2"
              strokeDasharray="4 4"
            />
          </svg>
        </motion.div>

        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-[#1a1a1a] border border-amber-500/30 p-4 rounded-xl relative"
        >
          <div className="flex items-center gap-2 mb-3 text-amber-400 text-xs font-mono uppercase">
            <AlignLeft className="w-4 h-4" /> Linked Caption
          </div>
          <div className="space-y-2">
            <div className="h-2 bg-white/20 rounded w-full" />
            <div className="h-2 bg-white/20 rounded w-4/5" />
          </div>
        </motion.div>
      </div>

      {/* Spatial Heuristics Engine */}
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="w-48 h-48 rounded-full border border-white/10 bg-white/5 flex flex-col items-center justify-center relative"
      >
        <div className="absolute inset-0 rounded-full border border-indigo-500/20 animate-[spin_10s_linear_infinite]" />
        <div className="absolute inset-4 rounded-full border border-indigo-500/20 animate-[spin_15s_linear_infinite_reverse]" />
        <Link className="w-8 h-8 text-indigo-400 mb-2" />
        <span className="text-xs font-mono text-gray-400 text-center px-4">Spatial<br/>Heuristics</span>
      </motion.div>
    </motion.div>
  );
}

function VLMSynthesisView() {
  const jsonOutput = `{
  "page_number": 4,
  "elements": {
    "figures": [
      {
        "id": "Figure 3",
        "caption": "Performance comparison across datasets.",
        "visual_description": "A bar chart showing our model outperforming baselines by 15% on CIFAR-10.",
        "scientific_insight": "Demonstrates the efficacy of the novel attention mechanism in low-data regimes.",
        "linked_text": ["text_block_4"]
      }
    ]
  }
}`;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="relative z-10 w-full max-w-3xl flex flex-col items-center"
    >
      <div className="flex items-center gap-4 mb-6">
        <div className="px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-gray-400 flex items-center gap-2">
          <ImageIcon className="w-3 h-3" /> Cropped Image
        </div>
        <span className="text-gray-600">+</span>
        <div className="px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-gray-400 flex items-center gap-2">
          <AlignLeft className="w-3 h-3" /> Text Blocks
        </div>
        <ArrowRight className="w-4 h-4 text-indigo-400 mx-2" />
        <div className="px-4 py-2 rounded-full bg-indigo-500/20 border border-indigo-500/30 text-xs font-mono text-indigo-300 flex items-center gap-2 shadow-[0_0_15px_rgba(99,102,241,0.2)]">
          <BrainCircuit className="w-3 h-3" /> Gemini 1.5 Pro
        </div>
      </div>

      <motion.div 
        initial={{ height: 0, opacity: 0 }}
        animate={{ height: 'auto', opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.5 }}
        className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl overflow-hidden"
      >
        <div className="bg-white/5 px-4 py-2 border-b border-white/10 flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500/50" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
          <span className="ml-2 text-[10px] font-mono text-gray-500">output.json</span>
        </div>
        <div className="p-4 overflow-x-auto">
          <pre className="text-xs font-mono leading-relaxed">
            <code className="text-emerald-400">
              {jsonOutput.split('\n').map((line, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 + (i * 0.05) }}
                >
                  {line}
                </motion.div>
              ))}
            </code>
          </pre>
        </div>
      </motion.div>
    </motion.div>
  );
}

function GraphIndexingView() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="relative z-10 w-full max-w-2xl"
    >
      {/* Query Input */}
      <motion.div 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mb-12 relative"
      >
        <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-indigo-400" />
        </div>
        <input 
          type="text" 
          readOnly
          value="What does Figure 3 show about performance?"
          className="w-full bg-white/5 border border-indigo-500/30 rounded-xl py-4 pl-12 pr-4 text-sm text-white shadow-[0_0_20px_rgba(99,102,241,0.1)] focus:outline-none"
        />
      </motion.div>

      {/* Graph Structure */}
      <div className="relative h-64 flex items-center justify-center">
        {/* Central Node */}
        <motion.div 
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, type: "spring" }}
          className="absolute z-20 w-24 h-24 bg-indigo-500/20 border border-indigo-500/50 rounded-full flex flex-col items-center justify-center shadow-[0_0_30px_rgba(99,102,241,0.3)]"
        >
          <ImageIcon className="w-6 h-6 text-indigo-300 mb-1" />
          <span className="text-[10px] font-mono text-indigo-200">Figure 3</span>
        </motion.div>

        {/* Connected Nodes */}
        <motion.div 
          initial={{ scale: 0, x: 0, y: 0 }}
          animate={{ scale: 1, x: -120, y: -60 }}
          transition={{ delay: 0.7, type: "spring" }}
          className="absolute z-10 w-32 p-3 bg-white/5 border border-white/10 rounded-lg"
        >
          <div className="text-[10px] font-mono text-emerald-400 mb-1">VLM Insight</div>
          <div className="text-xs text-gray-400 leading-tight">"Outperforms baselines by 15%..."</div>
        </motion.div>

        <motion.div 
          initial={{ scale: 0, x: 0, y: 0 }}
          animate={{ scale: 1, x: 120, y: -60 }}
          transition={{ delay: 0.8, type: "spring" }}
          className="absolute z-10 w-32 p-3 bg-white/5 border border-white/10 rounded-lg"
        >
          <div className="text-[10px] font-mono text-amber-400 mb-1">Caption</div>
          <div className="text-xs text-gray-400 leading-tight">"Performance comparison across..."</div>
        </motion.div>

        <motion.div 
          initial={{ scale: 0, x: 0, y: 0 }}
          animate={{ scale: 1, x: 0, y: 100 }}
          transition={{ delay: 0.9, type: "spring" }}
          className="absolute z-10 w-40 p-3 bg-white/5 border border-white/10 rounded-lg"
        >
          <div className="text-[10px] font-mono text-blue-400 mb-1">Source Text (Page 4)</div>
          <div className="text-xs text-gray-400 leading-tight">"As shown in Fig 3, our model..."</div>
        </motion.div>

        {/* Connecting Lines */}
        <svg className="absolute inset-0 w-full h-full -z-10">
          <motion.line initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1 }} x1="50%" y1="50%" x2="calc(50% - 120px)" y2="calc(50% - 60px)" stroke="rgba(255,255,255,0.1)" strokeWidth="2" strokeDasharray="4 4" />
          <motion.line initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1.1 }} x1="50%" y1="50%" x2="calc(50% + 120px)" y2="calc(50% - 60px)" stroke="rgba(255,255,255,0.1)" strokeWidth="2" strokeDasharray="4 4" />
          <motion.line initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1.2 }} x1="50%" y1="50%" x2="50%" y2="calc(50% + 100px)" stroke="rgba(255,255,255,0.1)" strokeWidth="2" strokeDasharray="4 4" />
        </svg>
      </div>
    </motion.div>
  );
}
