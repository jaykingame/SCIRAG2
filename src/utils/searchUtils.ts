export function tokenize(text: string): string[] {
  if (!text) return [];
  // Remove punctuation and convert to lowercase
  const cleanText = text.toLowerCase().replace(/[^\w\s]/g, ' ');
  return cleanText.split(/\s+/).filter(word => word.length > 2);
}

export function calculateKeywordScore(queryTokens: string[], textTokens: string[]): number {
  if (queryTokens.length === 0 || textTokens.length === 0) return 0;
  const textSet = new Set(textTokens);
  let matchCount = 0;
  for (const token of queryTokens) {
    if (textSet.has(token)) matchCount++;
  }
  return matchCount / queryTokens.length;
}

export function reciprocalRankFusion<T>(rankedLists: T[][], k = 60): { item: T, score: number }[] {
  const scores = new Map<T, number>();
  
  for (const list of rankedLists) {
    list.forEach((item, index) => {
      const rank = index + 1;
      const currentScore = scores.get(item) || 0;
      scores.set(item, currentScore + 1 / (k + rank));
    });
  }

  return Array.from(scores.entries())
    .map(([item, score]) => ({ item, score }))
    .sort((a, b) => b.score - a.score);
}
