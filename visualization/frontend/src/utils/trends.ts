/** Remember last numeric values and emit ↑ ↓ → for trend display. */
const prev = new Map<string, number>();

export function trendArrow(key: string, value: number): string {
  const last = prev.get(key);
  prev.set(key, value);
  if (last === undefined) return "→";
  if (value > last + 1e-9) return "↑";
  if (value < last - 1e-9) return "↓";
  return "→";
}
