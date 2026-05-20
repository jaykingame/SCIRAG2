import { GoogleGenAI, Type } from "@google/genai";

// Initialize the Gemini SDK
// The API key is injected by the AI Studio environment
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export interface ParsedPageResult {
  page_number: number;
  figures: {
    id: string;
    caption: string;
    visual_description: string;
    scientific_insight: string;
    linked_text_quotes: string[];
    bounding_box_2d?: number[];
  }[];
  tables: {
    id: string;
    caption: string;
    key_findings: string;
    linked_text_quotes: string[];
    bounding_box_2d?: number[];
  }[];
  key_claims: string[];
}

export const PARSER_MODELS = [
  "gemini-3.1-flash-lite-preview",
  "gemini-3-flash-preview",
  "gemini-3.1-pro-preview",
  "gemini-2.5-flash-native-audio-preview-12-2025"
];

export async function parsePageWithGemini(base64Image: string, pageNum: number, rawText: string | null = null, modelIndex = 0): Promise<ParsedPageResult> {
  const modelName = PARSER_MODELS[modelIndex % PARSER_MODELS.length];
  
  const textContextBlock = rawText ? `
  RAW EXTRACTED TEXT (Note: May have out-of-order columns due to PDF parsing natively reading horizontally. Use the visual image to figure out the correct semantic flow and column ordering, but you can use this raw text to ensure you spell specific terms and numbers exactly correctly!):
  ---
  ${rawText}
  ---
  ` : '';

  const prompt = `You are a scientific multimodal reasoning system.
  I am providing an image of page ${pageNum} from a research paper.
  ${textContextBlock}
  
  YOUR TASK:
  Extract and structure the scientific meaning of this page.
  
  INSTRUCTIONS:
  1. Important: You must read the document naturally (vertically down columns, respecting the spatial layout shown in the image). Do not just copy the raw text if it reads horizontally across two columns causing gibberish.
  2. Identify all figures and tables on this page.
  3. Link them to their captions.
  4. Extract the scientific insight (what the figure/table means) and any surrounding text quotes that reference them.
  5. Extract the main scientific claims, methods, or results discussed in the text on this page.
  6. For each figure and table, provide its bounding_box_2d in the format [ymin, xmin, ymax, xmax] where coordinates are scaled to 1000 (e.g., [100, 100, 500, 500]).
  7. Do not hallucinate. If there are no figures or tables, leave those arrays empty.`;

  const response = await ai.models.generateContent({
    model: modelName,
    contents: {
      parts: [
        { text: prompt },
        { inlineData: { mimeType: "image/jpeg", data: base64Image } }
      ]
    },
    config: {
      responseMimeType: "application/json",
      temperature: 0.1,
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          page_number: { type: Type.INTEGER, description: "The page number" },
          figures: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING, description: "e.g., Figure 1" },
                caption: { type: Type.STRING },
                visual_description: { type: Type.STRING, description: "What is visually shown in the figure" },
                scientific_insight: { type: Type.STRING, description: "The scientific meaning or takeaway of the figure" },
                linked_text_quotes: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Exact quotes from the text referencing this figure" },
                bounding_box_2d: {
                  type: Type.ARRAY,
                  items: { type: Type.INTEGER },
                  description: "Bounding box of the figure in [ymin, xmin, ymax, xmax] format, coordinates scaled to 1000"
                }
              }
            }
          },
          tables: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING, description: "e.g., Table 1" },
                caption: { type: Type.STRING },
                key_findings: { type: Type.STRING, description: "Summary of the data or findings in the table" },
                linked_text_quotes: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Exact quotes from the text referencing this table" },
                bounding_box_2d: {
                  type: Type.ARRAY,
                  items: { type: Type.INTEGER },
                  description: "Bounding box of the table in [ymin, xmin, ymax, xmax] format, coordinates scaled to 1000"
                }
              }
            }
          },
          key_claims: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "Main scientific claims, methods, or results discussed in the text on this page"
          }
        }
      }
    }
  });

  if (!response.text) {
    throw new Error("No response received from Gemini.");
  }

  return JSON.parse(response.text) as ParsedPageResult;
}

export interface IndexItem {
  id: string;
  type: 'text' | 'figure' | 'table' | 'page';
  pageNumber: number;
  text: string;
  imageBase64?: string;
  embedding: number[];
  boundingBox?: number[];
}

export async function generateEmbeddings(contents: any[]) {
  const result = await ai.models.embedContent({
    model: 'gemini-embedding-2-preview',
    contents: contents,
  });
  return result.embeddings;
}

export async function askDocument(question: string, contextItems: IndexItem[], modelIndex = 0) {
  const modelName = PARSER_MODELS[modelIndex % PARSER_MODELS.length];
  const parts: any[] = [];
  parts.push({ text: `You are a scientific assistant. 

CRITICAL GROUNDING RULES:
1. Answer the user's question based ONLY on the provided context items.
2. DO NOT use any external knowledge about papers, games, or scientific topics.
3. If the answer is not in the context, explicitly state: "I cannot find information about [topic] in the provided context."
4. You MUST cite your sources inline using the format [Page X, Type].
5. Match figure/table references flexibly (e.g., "Fig 1", "Fig. 1", and "Figure 1" refer to the exact same thing). Do not claim a figure is missing if it is present under a slightly different abbreviation.

Context Items:\n` });

  contextItems.forEach((item, i) => {
    parts.push({ text: `\n--- Context Item ${i + 1} (${item.type.toUpperCase()} from Page ${item.pageNumber}) ---\n` });
    if (item.text) {
      parts.push({ text: item.text + "\n" });
    }
    if (item.imageBase64) {
      parts.push({ inlineData: { mimeType: "image/jpeg", data: item.imageBase64 } });
    }
  });

  parts.push({ text: `\n\nQuestion: ${question}` });

  const response = await ai.models.generateContent({
    model: modelName,
    contents: { parts },
  });

  return response.text;
}

export async function* askDocumentStream(question: string, contextItems: IndexItem[], chatHistory: {role: 'user' | 'assistant', text: string}[] = [], abortSignal?: AbortSignal, modelIndex = 0) {
  const modelName = PARSER_MODELS[modelIndex % PARSER_MODELS.length];
  const parts: any[] = [];
  parts.push({ text: `You are a scientific assistant. 

CRITICAL GROUNDING RULES:
1. Answer the user's question based ONLY on the provided context items.
2. DO NOT use any external knowledge about papers, games, or scientific topics.
3. If the answer is not in the context, explicitly state: "I cannot find information about [topic] in the provided context."
4. You MUST cite your sources inline using the format [Page X, Type] where Type is Figure, Table, or Text. For example: 'The new architecture achieves a 15% speedup [Page 4, Table].'
5. Match figure/table references flexibly (e.g., "Fig 1", "Fig. 1", and "Figure 1" refer to the exact same thing). Do not claim a figure is missing if it is present under a slightly different abbreviation.

Context Items:\n` });

  contextItems.forEach((item, i) => {
    parts.push({ text: `\n--- Context Item ${i + 1} (${item.type.toUpperCase()} from Page ${item.pageNumber}) ---\n` });
    if (item.text) {
      parts.push({ text: item.text + "\n" });
    }
    if (item.imageBase64) {
      parts.push({ inlineData: { mimeType: "image/jpeg", data: item.imageBase64 } });
    }
  });

  // Add chat history for context
  if (chatHistory.length > 0) {
    parts.push({ text: `\n\n--- Chat History ---\n` });
    chatHistory.forEach(msg => {
      parts.push({ text: `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.text}\n` });
    });
  }

  parts.push({ text: `\n\nQuestion: ${question}` });

  const responseStream = await ai.models.generateContentStream({
    model: modelName,
    contents: { parts },
  });

  for await (const chunk of responseStream) {
    if (abortSignal?.aborted) {
      break;
    }
    if (chunk.text) {
      yield chunk.text;
    }
  }
}

export async function rewriteQuery(question: string, chatHistory: {role: 'user' | 'assistant', text: string}[], modelIndex = 0): Promise<string> {
  const modelName = PARSER_MODELS[modelIndex % PARSER_MODELS.length];
  if (chatHistory.length === 0) return question;

  const historyText = chatHistory.map(msg => `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.text}`).join('\n');
  const prompt = `You are a query rewriting assistant for a scientific document search engine.
Given the following chat history and a new user question, rewrite the new question so that it is a standalone, fully-contextualized search query.
If the new question refers to "it", "this", "the method", etc., replace those pronouns with the actual subject from the history.
If the new question is already standalone, just return it as is.
Return ONLY the rewritten query string, nothing else.

--- Chat History ---
${historyText}

--- New Question ---
${question}

Rewritten Query:`;

  const response = await ai.models.generateContent({
    model: modelName,
    contents: prompt,
    config: {
      temperature: 0.1,
    }
  });

  return response.text?.trim() || question;
}

export async function generateReport(parsedPages: any[], modelIndex = 0): Promise<string> {
  const modelName = PARSER_MODELS[modelIndex % PARSER_MODELS.length];
  
  // Aggregate content
  let aggregatedText = "";
  parsedPages.forEach(page => {
    aggregatedText += `\n--- Page ${page.result.page_number} ---\n`;
    if (page.result.key_claims && page.result.key_claims.length > 0) {
      aggregatedText += "Key Claims:\n" + page.result.key_claims.map((c: string) => `- ${c}`).join("\n") + "\n";
    }
    if (page.result.figures && page.result.figures.length > 0) {
      aggregatedText += "Figures:\n" + page.result.figures.map((f: any) => `- ${f.id}: ${f.caption}. Insight: ${f.scientific_insight}`).join("\n") + "\n";
    }
    if (page.result.tables && page.result.tables.length > 0) {
      aggregatedText += "Tables:\n" + page.result.tables.map((t: any) => `- ${t.id}: ${t.caption}. Findings: ${t.key_findings}`).join("\n") + "\n";
    }
  });

  const prompt = `You are an expert scientific researcher and writer. 
I am providing you with the extracted data (key claims, figures, tables) from a parsed research paper.

YOUR TASK:
Generate a comprehensive, well-structured scientific report summarizing the entire paper based ONLY on the provided extracted data.

The report MUST include the following sections (use Markdown headings):
# Abstract
# Introduction & Literature Review
# Methodology
# Results & Key Findings (reference the figures and tables if applicable)
# Conclusion

Extracted Data:
${aggregatedText}

Write the report in a professional, academic tone. Do not hallucinate information not present in the extracted data.`;

  const response = await ai.models.generateContent({
    model: modelName,
    contents: prompt,
  });

  return response.text || "Failed to generate report.";
}

export async function expandQuery(question: string, modelIndex = 0): Promise<string[]> {
  const modelName = PARSER_MODELS[modelIndex % PARSER_MODELS.length];
  const prompt = `You are a scientific search assistant. Given a user's question about a research paper, generate up to 3 distinct search queries (including the original if good) that would help retrieve the most relevant text, figures, and tables. Return ONLY a JSON array of strings.
  Question: ${question}`;

  const response = await ai.models.generateContent({
    model: modelName,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      temperature: 0.2,
    }
  });

  try {
    return JSON.parse(response.text || '[]');
  } catch (e) {
    return [question];
  }
}
