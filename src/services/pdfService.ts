import * as pdfjsLib from 'pdfjs-dist';
// @ts-ignore
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Use local worker to avoid CORS/CDN issues
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

export async function extractPagesAndText(file: File, maxPages: number = 50): Promise<{ images: string[], texts: string[], totalPages: number }> {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const numPages = pdf.numPages;
  const images: string[] = [];
  const texts: string[] = [];

  const pagesToProcess = Math.min(numPages, maxPages);

  for (let i = 1; i <= pagesToProcess; i++) {
    const page = await pdf.getPage(i);
    
    // Extract raw text (Simulating PyPDF2 out-of-order column reading)
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map((item: any) => item.str).join(' ');
    texts.push(pageText);

    // Render Canvas for VLM (Preserves column layout spatially)
    const viewport = page.getViewport({ scale: 2.0 }); 
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    if (!context) throw new Error("Could not get canvas context");

    canvas.height = viewport.height;
    canvas.width = viewport.width;

    await page.render({ canvasContext: context, viewport: viewport } as any).promise;
    
    const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    images.push(base64);
  }
  
  return { images, texts, totalPages: numPages };
}
