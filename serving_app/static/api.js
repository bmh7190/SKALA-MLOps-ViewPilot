// API 호출과 오류 처리를 한곳에 둔다. 화면 디자인을 바꿔도 이 모듈은 재사용한다.
async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 600_000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch { data = null; }
    if (!response.ok) {
      const detail = data?.detail;
      const message = Array.isArray(detail)
        ? detail.map(item => `${item.loc?.join(".") || "입력"}: ${item.msg}`).join(" / ")
        : typeof detail === "string" ? detail : `서버 응답 오류 (${response.status})`;
      throw new Error(message);
    }
    if (data === null) throw new Error("서버가 올바른 JSON을 반환하지 않았습니다.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("응답 대기 시간을 초과했습니다. 서버 작업은 계속될 수 있으니 상태와 로그를 새로고침하세요.");
    throw error;
  } finally { clearTimeout(timer); }
}

function query(params) {
  return new URLSearchParams(Object.entries(params).filter(([, value]) => value !== "" && value != null)).toString();
}
function jsonPost(path, body) {
  return request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
function upload(path, file) {
  const body = new FormData();
  body.append("file", file);
  return request(path, { method: "POST", body });
}

export const api = {
  health: () => request("/health"),
  predictions: params => request(`/api/v1/videos/predictions?${query(params)}`),
  drift: params => request(`/api/v1/monitoring/drift?${query(params)}`),
  scenario: name => request(`/api/v1/simulations/${name}/run`, { method: "POST" }),
  log: () => request("/api/v1/logs/aiops.log"),
  dataset: purpose => request(`/api/v1/datasets/latest?${query({ purpose })}`),
  uploadDataset: (purpose, file) => upload(`/api/v1/datasets?${query({ purpose })}`, file),
  predictCsv: file => upload("/api/v1/videos/predictions/csv", file),
  actual: body => jsonPost("/api/v1/videos/actuals", body),
  train: body => jsonPost("/api/v1/models/training", body),
};
