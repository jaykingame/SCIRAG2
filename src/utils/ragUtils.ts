export async function cropImage(base64Image: string, boundingBox: number[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const [ymin, xmin, ymax, xmax] = boundingBox;
      const top = (ymin / 1000) * img.naturalHeight;
      const left = (xmin / 1000) * img.naturalWidth;
      const bottom = (ymax / 1000) * img.naturalHeight;
      const right = (xmax / 1000) * img.naturalWidth;
      const width = Math.max(right - left, 1);
      const height = Math.max(bottom - top, 1);

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('No canvas context'));
        return;
      }

      ctx.drawImage(img, left, top, width, height, 0, 0, width, height);
      const croppedBase64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
      resolve(croppedBase64);
    };
    img.onerror = () => reject(new Error('Image load error'));
    img.src = `data:image/jpeg;base64,${base64Image}`;
  });
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
