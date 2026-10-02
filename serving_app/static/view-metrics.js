// Presentation-only helpers. Server drift decisions are never recomputed here.
export function errorMetrics(row) {
  if (row.target_views_day7 == null) return null;
  const predicted = Number(row.predicted_views_day7);
  const actual = Number(row.target_views_day7);
  if (!Number.isFinite(predicted) || !Number.isFinite(actual) || predicted < 0 || actual < 0) return null;
  const ratio = (predicted + 1) / (actual + 1);
  return { delta: predicted - actual, ratio, magnitude: Math.abs(Math.log(ratio)),
    review: ratio >= 2 || ratio <= 0.5 };
}

export function selectVideos(rows, filter, sort) {
  const selected = rows.filter(row => {
    const m = errorMetrics(row);
    return filter === "review" ? !!m?.review : filter === "observed" ? m !== null :
      filter === "pending" ? row.target_views_day7 == null : true;
  });
  if (sort !== "recent") selected.sort((a, b) => {
    const x = errorMetrics(a), y = errorMetrics(b);
    const rank = m => m == null ? -1 : sort === "absolute" ? Math.abs(m.delta) : m.magnitude;
    return rank(y) - rank(x);
  });
  return selected;
}

export function relatedBlocks(videos, blocks) {
  return blocks.filter(block => videos.some(video => video.block_id != null &&
    Number(video.block_id) === Number(block.block_id) &&
    String(video.model_version) === String(block.model_version) && video.mode === block.mode));
}

export async function fetchAllPages(fetchPage, params = {}) {
  const items = [];
  let total = 0;
  for (let page = 1; ; page++) {
    const result = await fetchPage({...params, page, page_size: 100});
    if (!Array.isArray(result.items) || !Number.isFinite(result.total)) throw new Error('이력 응답 형식 오류');
    total = result.total;
    items.push(...result.items);
    if (items.length >= total) return {items, total};
    if (!result.items.length || page >= 1000) throw new Error('전체 이력을 읽지 못했습니다. 모델 버전 필터로 범위를 줄여주세요.');
  }
}
