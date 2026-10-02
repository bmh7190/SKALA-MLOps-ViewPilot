import { api } from "./api.js";
import { errorMetrics, selectVideos, relatedBlocks, fetchAllPages } from "./view-metrics.js";

const $ = id => document.getElementById(id);
const state = {
  health: null, initialized: false, busy: false, completed: new Set(),
  predictionPage: 1, driftPage: 1, predictionTotal: 0, driftTotal: 0,
  videos: [], blocks: [], reviewVideo: null,
};
const operations = new URLSearchParams(location.search).get('view') === 'operations';
let displayMetadata = {};
const categoryNames = {gaming:'게임',education:'교육',entertainment:'엔터테인먼트',lifestyle:'라이프스타일'};
document.body.classList.toggle('operations-page', operations);
if (operations) {
  document.querySelector('h1').textContent = '데이터 관리';
  document.querySelector('.page-heading .muted').textContent = 'CSV 예측, 학습 및 운영 로그를 관리합니다.';
}
const pageSizes = { prediction: 20, drift: 100 };
const format = value => value == null ? "—" : Number(value).toLocaleString("ko-KR");
const score = value => value == null ? "—" : Number(value).toFixed(4);
const modeName = mode => mode === "live" ? "운영" : "시뮬레이션";
const statusNames = { ok: "정상", warning: "주의", retrain_review: "재학습 검토", collecting: "수집 중", baseline_required: "기준값 필요" };

function notice(message, level = "") {
  $("notice").textContent = message;
  $("notice").className = `notice ${level}`;
}
function element(tag, text, className = "") {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function badge(status) {
  return element("span", statusNames[status] || status, `badge ${status === "ok" ? "ok" : "warn"}`);
}
function filters() {
  return { mode: $("history-mode").value, model_version: $("history-version").value.trim() };
}

// 서버 준비 상태와 실행 잠금을 함께 반영한다. 실행 중에는 모든 쓰기 요청을 막는다.
function updateControls() {
  document.querySelectorAll(".action-control").forEach(control => { control.disabled = state.busy; });
  const ready = state.health?.model_ready && state.health?.demo_mode && state.health?.baseline_rmsle != null;
  for (const name of ["normal", "drift"]) {
    $(`run-${name}`).disabled = state.busy || !ready || state.completed.has(name);
  }
  $("predict-csv").disabled = state.busy || !state.health?.model_ready;
  for (const kind of ["prediction", "drift"]) {
    const page = state[`${kind}Page`];
    $(`${kind}-prev`).disabled = state.busy || page <= 1;
    $(`${kind}-next`).disabled = state.busy || page * pageSizes[kind] >= state[`${kind}Total`];
  }
  document.body.setAttribute("aria-busy", String(state.busy));
}

// 경과 시간만 표시한다. 서버가 주지 않는 단계별 진행률을 만들어 표시하지 않는다.
async function action(label, work) {
  if (state.busy) return;
  state.busy = true;
  const started = Date.now();
  const showElapsed = () => { $("busy-status").textContent = `${label} · ${Math.floor((Date.now() - started) / 1000)}초 경과 · 완료 후 결과가 표시됩니다.`; };
  showElapsed();
  const timer = setInterval(showElapsed, 1000);
  updateControls();
  try { await work(); }
  catch (error) { notice(error.message || "요청을 완료하지 못했습니다.", "error"); }
  finally {
    clearInterval(timer);
    $("busy-status").textContent = "";
    state.busy = false;
    updateControls();
  }
}

function renderHealth(health) {
  $("connection").textContent = "서버 연결됨";
  $("connection").className = "badge ok";
  $("model-version").textContent = health.model_ready ? `v${health.model_version}` : "모델 없음";
  $("model-mode").textContent = health.demo_mode ? "시연 모드 · 재시작 시 초기화" : "일반 모드 · 기록 보존";
  $("baseline").textContent = score(health.baseline_rmsle);
  $("threshold").textContent = score(health.drift_threshold);
  $("monitor-state").textContent = !health.model_ready ? "학습 필요" : health.baseline_rmsle == null ? "기준값 필요" : "평가 준비됨";
  $("monitor-description").textContent = "실제값이 확보된 30개 묶음으로 평가";
  $("scenario-help").textContent = !health.demo_mode
    ? "버튼 시연은 DEMO_MODE=true로 서버를 시작한 뒤 사용할 수 있습니다."
    : !health.model_ready ? "최초 모델 학습과 기준값 계산을 먼저 완료하세요."
    : health.baseline_rmsle == null ? "새 모델이 선택되었습니다. 다시 시연하려면 서버를 재시작하세요."
    : "정상 → 드리프트 순서 또는 각각 단독 실행할 수 있습니다. 같은 시나리오를 다시 실행하려면 서버를 재시작하세요.";
}

async function refreshOverview() {
  try {
    const health = await api.health();
    state.health = health;
    if (!state.initialized) {
      $("history-mode").value = "";
      state.initialized = true;
    }
    renderHealth(health);
  } catch (error) {
    state.health = null;
    $("connection").textContent = "서버 연결 확인 필요";
    $("connection").className = "badge error";
    for (const id of ["model-version", "baseline", "threshold", "monitor-state"]) $(id).textContent = "—";
    $("scenario-help").textContent = "서버 연결을 확인한 뒤 새로고침하세요.";
    notice(error.message, "error");
  }
  await Promise.all([loadHistory("prediction"), loadHistory("drift"), loadLog(), loadDataset(), loadQuality()]);
  // 서버가 시연 기록을 비운 경우 이전 페이지의 실행 결과도 제거한다.
  if (state.health?.demo_mode && filters().mode === "simulation" && !filters().model_version &&
      state.predictionTotal === 0 && state.driftTotal === 0) {
    state.completed.clear();
    $("scenario-result").hidden = true;
  }
  updateControls();
}

async function loadQuality() {
  const health = state.health;
  const banner = $("quality-banner");
  banner.className = "quality-banner";
  $("quality-status").className = "badge";
  if (!health?.model_ready || health.baseline_rmsle == null) {
    $("quality-title").textContent = !health ? "서버 연결 확인 필요" : !health.model_ready ? "예측 모델 준비 필요" : "새 모델의 기준값 설정 필요";
    $("quality-description").textContent = "평가 준비 전에는 예측 품질을 정상으로 표시하지 않습니다. 운영 모델 설정을 확인하세요.";
    $("quality-status").textContent = "평가 대기";
    return;
  }
  try {
    const mode = health.demo_mode ? "simulation" : "live";
    const data = await api.drift({mode, model_version: health.model_version, page: 1, page_size: 1});
    const last = data.items[0];
    const status = last?.status || "collecting";
    const messages = {
      ok: ["최근 예측 오차가 기준 이내입니다", "설정한 경보 조건에 해당하지 않습니다. 예측의 정확성을 보장하는 의미는 아닙니다."],
      warning: ["최근 예측 오차가 커졌습니다", "한 묶음에서 임계값을 초과했습니다. 다음 평가 묶음의 결과를 확인하세요."],
      retrain_review: ["예측 품질 저하 감지 · 모델 점검 필요", "두 묶음 이상 연속으로 기준을 초과했습니다. 예측값과 실제 성과를 함께 확인하세요. 재학습 결과는 실행 이력에서 별도로 확인합니다."],
      collecting: ["첫 평가 결과를 기다리고 있습니다", "실제 7일 결과가 확보된 영상 30개가 모이면 평가합니다."],
    };
    $("quality-title").textContent = messages[status][0];
    $("quality-description").textContent = `현재 모델 v${health.model_version} · ${modeName(mode)} · ${messages[status][1]}${last ? ` 최근 묶음 #${last.block_id}: RMSLE ${score(last.rmsle)} / 임계값 ${score(last.threshold)}` : ""}`;
    $("quality-status").textContent = status === "retrain_review" ? "드리프트 의심" : statusNames[status];
    const tone = status === "retrain_review" ? "error" : status === "warning" ? "warn" : status === "ok" ? "ok" : "";
    banner.className = `quality-banner ${tone}`;
    $("quality-status").className = `badge ${tone}`;
  } catch (error) {
    $("quality-title").textContent = "예측 품질 조회 실패";
    $("quality-description").textContent = error.message;
    $("quality-status").textContent = "확인 불가";
  }
}

function showVideo(row) {
  if ($('video-filter').value === 'review') { state.reviewVideo = row.prediction_id; state.driftPage = 1; $('drift-scope').value = 'related'; renderRelated(); }
  $("video-detail").hidden = false;
  // $("video-detail-title").textContent = row.video_id;
  $("video-detail-title").textContent = displayMetadata[row.video_id]?.source_video_id || row.video_id;
  const m = errorMetrics(row);
  const metadata = displayMetadata[row.video_id] || {};
  $("video-detail-content").replaceChildren(
    element("p", `예측 기록: ${row.prediction_id} · 모델 v${row.model_version} · ${modeName(row.mode)}`),
    element("p", `예측 시각: ${new Date(row.predicted_at).toLocaleString("ko-KR")}`),
    element("p", m ? `예측 ${format(row.predicted_views_day7)}회 / 실제 ${format(row.target_views_day7)}회 · ${m.delta > 0 ? "과대예측" : m.delta < 0 ? "과소예측" : "일치"}` : "실제 결과가 등록되면 오차를 계산합니다."),
    element("p", metadata.category ? `카테고리: ${categoryNames[metadata.category] || metadata.category} · 게시 당시 구독자 ${format(metadata.subscriber_count_at_publish)}명 · 출처: ${metadata.source}` : '추가 메타데이터가 없습니다.', 'hint'),
    element('p',metadata.ui_demo ? `원본 ${metadata.source_video_id}의 화면 시연용 복제본입니다. 정답은 원본 CSV에 보존되어 있으며 독립 성능 평가에 사용하지 않습니다.` : '이미지는 카테고리별 데모 디자인이며 실제 영상 썸네일이 아닙니다.','hint'),
  );
}

function renderVideos() {
  const rows = state.videos;
  const observed = rows.filter(row => errorMetrics(row) !== null);
  $("video-total").textContent = format(rows.length);
  $("video-observed").textContent = `${observed.length}건`;
  $("video-review").textContent = `${observed.filter(row => errorMetrics(row).review).length}건`;
  $("video-pending").textContent = `${rows.filter(row => row.target_views_day7 == null).length}건`;
  const filter = $('video-filter').value;
  $('history-description').hidden = filter === 'pending';
  const selected = selectVideos(rows, filter, $('video-sort').value);
  state.predictionTotal = selected.length;
  state.predictionPage = Math.min(state.predictionPage, Math.max(1, Math.ceil(selected.length / pageSizes.prediction)));
  const visible = selected.slice((state.predictionPage-1)*pageSizes.prediction, state.predictionPage*pageSizes.prediction);
  $('prediction-page-label').textContent = `${selected.length}건 · ${state.predictionPage} / ${Math.max(1,Math.ceil(selected.length/pageSizes.prediction))}페이지`;
  $('history-heading').textContent = {pending:'결과 대기 영상 · 7일 예측',observed:'실제 결과 확보 · 예측과 비교',review:'큰 오차 영상 · 사후 점검'}[filter];
  document.querySelectorAll('[data-video-view]').forEach(button => button.setAttribute('aria-pressed',String(button.dataset.videoView===filter)));
  document.querySelector('.video-table').classList.toggle('pending-view',filter==='pending');
  renderTable("prediction-rows", visible, [
    row => {
      const cell = element("div");
      cell.className = 'video-identity';
      const metadata = displayMetadata[row.video_id] || {};
      const cat = row.category || metadata.category;
      const thumb = categoryNames[cat] ? element('img') : element('span', '이미지 없음', 'thumbnail-placeholder');
      if (categoryNames[cat]) { thumb.src=`/demo-thumbnails/${cat}.svg`; thumb.alt=`${categoryNames[cat]} 데모용 썸네일 (실제 영상 이미지 아님)`; thumb.className='demo-thumbnail'; thumb.loading='lazy'; }
      cell.append(thumb);
      const info = element('div');
      const displayId = metadata.source_video_id || row.video_id;
      const button = element("button", displayId, "video-link");
      // const button = element("button", row.video_id, "video-link"); button.type = "button";
      button.addEventListener("click", () => showVideo(row));
      const subscribers = row.subscriber_count_at_publish ?? metadata.subscriber_count_at_publish;
      info.append(button, element("small", `${categoryNames[cat] || cat || '카테고리 미제공'} · ${subscribers == null ? '구독자 수 미제공' : `구독자 ${format(subscribers)}명`}`), element('small',`v${row.model_version} · 예측 ${new Date(row.predicted_at).toLocaleString('ko-KR')}`), element('small',metadata.ui_demo ? 'UI 데모 복제본 · 독립 평가용 아님' : `${modeName(row.mode)} · 이미지는 데모 디자인`));
      cell.append(info);
      return cell;
    },
    row => format(row.predicted_views_day7),
    row => row.target_views_day7 == null ? "결과 대기" : format(row.target_views_day7),
    row => { const m = errorMetrics(row); return m ? `${m.delta > 0 ? "+" : ""}${format(m.delta)}` : "—"; },
    row => { const m = errorMetrics(row); return m ? `${m.ratio.toFixed(2)}배` : "—"; },
    row => { const m = errorMetrics(row); return element("span", !m ? "결과 대기" : m.review ? `큰 오차 · ${m.delta > 0 ? "과대" : "과소"}` : "결과 확보", `badge ${m?.review ? "warn" : ""}`); },
    actualButton,
  ], filter==='pending' ? '실제값 미등록 영상이 없습니다. 데이터 관리에서 정답 없는 CSV로 새 영상을 예측하세요. 시나리오 결과는 이미 실제값이 포함될 수 있습니다.' : '이 조건에 해당하는 기록이 없습니다.');
  renderRelated();
}

function renderRelated() {
  const review = $('video-filter').value === 'review';
  $('related-drift').hidden = !review;
  if (!review) return;
  $('related-drift').open = true;
  let videos = state.videos.filter(row => errorMetrics(row)?.review);
  if (state.reviewVideo) videos = videos.filter(row => row.prediction_id === state.reviewVideo);
  const all = $('drift-scope').value === 'all';
  const blocks = all ? state.blocks : relatedBlocks(videos,state.blocks);
  state.driftTotal = blocks.length;
  state.driftPage = Math.min(state.driftPage,Math.max(1,Math.ceil(blocks.length/pageSizes.drift)));
  const page = blocks.slice((state.driftPage-1)*pageSizes.drift,state.driftPage*pageSizes.drift);
  $('drift-heading').textContent = all ? '정상부터 드리프트까지 · 전체 평가 흐름' : '영상이 포함된 묶음의 평가';
  $('related-description').textContent = all
    ? `현재 데이터 구분·모델 버전 조회 조건에 맞는 전체 ${blocks.length}개 묶음입니다. 정상 평가도 포함합니다. 버전이나 운영/시뮬레이션이 섞이면 하나의 연속 드리프트 판정으로 해석하지 마세요. 필요하면 위 조회 필터를 지정하세요.`
    : `${state.reviewVideo ? '선택한 영상' : '큰 오차 영상 전체'}에 연결된 ${blocks.length}개 평가 묶음입니다. RMSLE는 영상 한 개가 아닌 30개 묶음의 오차입니다. 영상 ID를 누르면 해당 기록의 묶음만 표시합니다.`;
  const empty = all ? '현재 조회 조건에 완료된 평가 묶음이 없습니다.' : '연결된 평가 묶음이 없습니다. 실제값이 있어도 평가 대기 중이거나 아직 묶음에 포함되지 않았을 수 있습니다.';
  renderTable('drift-rows',page,[r=>`#${r.block_id}`,r=>score(r.rmsle),r=>score(r.threshold),r=>badge(r.status),r=>`${r.consecutive_exceeds}회`,r=>`v${r.model_version}`,r=>modeName(r.mode)],empty);
  renderChart(page);
  $('chart-empty').textContent = empty;
  $('drift-page-label').textContent = `${blocks.length}개 묶음 · ${state.driftPage}페이지`;
}

function renderTable(bodyId, rows, columns, emptyMessage) {
  const body = $(bodyId);
  body.replaceChildren();
  if (!rows.length) {
    const cell = element("td", emptyMessage, "empty");
    cell.colSpan = columns.length;
    const row = element("tr"); row.append(cell); body.append(row);
    return;
  }
  for (const item of rows) {
    const row = element("tr");
    for (const column of columns) {
      const cell = element("td");
      const value = column(item);
      if (value instanceof Node) cell.append(value);
      else cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
}

function actualButton(row) {
  if (displayMetadata[row.video_id]?.ui_demo) return '-';
  if (row.mode !== "live" || row.target_views_day7 != null) return "—";
  const button = element("button", "실제값 입력", "action-control secondary");
  button.type = "button";
  button.addEventListener("click", () => {
    $("actual-id").value = row.prediction_id;
    $("actual-value").value = "";
    $("actual-form").hidden = false;
    $("actual-value").focus();
  });
  return button;
}

async function loadHistory(kind) {
  const errorBox = $(`${kind}-error`);
  errorBox.textContent = "";
  try {
    let data = await fetchAllPages(kind === 'prediction' ? api.predictions : api.drift, filters());
    data.page = 1; data.page_size = 100;
    state[`${kind}Total`] = data.total;
    $(`${kind}-page-label`).textContent = `${format(data.total)}건 · ${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))}페이지`;
    if (kind === "prediction") {
      state.videos = data.items;
      $("video-detail").hidden = true;
      renderVideos();
    } else {
      state.blocks = data.items;
      renderTable("drift-rows", data.items, [
        row => `#${row.block_id}`, row => score(row.rmsle), row => score(row.threshold),
        row => badge(row.status), row => `${row.consecutive_exceeds}회`,
        row => `v${row.model_version}`, row => modeName(row.mode),
      ], "완료된 평가 묶음이 없습니다. 실제값 30개가 모이면 평가합니다.");
      renderChart(data.items);
      renderRelated();
    }
  } catch (error) {
    errorBox.textContent = `이력 조회 실패: ${error.message}`;
    $(`${kind}-rows`).replaceChildren();
    $(`${kind}-page-label`).textContent = "불러오기 실패";
    if (kind === "drift") { state.blocks = []; renderChart([]); renderRelated(); }
    else {
      state.videos = [];
      $("video-detail").hidden = true;
      for (const id of ["video-total", "video-observed", "video-review", "video-pending"]) $(id).textContent = "—";
    }
  }
  updateControls();
}

// SVG에는 서버의 숫자만 좌표로 넣고, 문자열은 textContent로 표시한다.
function renderChart(items) {
  const svg = $("drift-chart");
  const rows = [...items].sort((a, b) => a.block_id - b.block_id);
  svg.replaceChildren();
  svg.toggleAttribute("hidden", !rows.length);
  $("chart-empty").hidden = !!rows.length;
  if (!rows.length) return;
  const make = (tag, attributes, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text != null) node.textContent = text;
    svg.append(node);
    return node;
  };
  const top = Math.max(0.1, ...rows.flatMap(row => [row.rmsle, row.threshold])) * 1.15;
  const x = index => rows.length === 1 ? 422 : 56 + index / (rows.length - 1) * 730;
  const y = value => 198 - value / top * 176;
  for (let tick = 0; tick <= 4; tick++) {
    const value = top * tick / 4;
    make("line", { x1: 56, x2: 786, y1: y(value), y2: y(value), stroke: "#303949" });
    make("text", { x: 46, y: y(value) + 4, "text-anchor": "end" }, value.toFixed(2));
  }
  for (const [field, color, dashed] of [["rmsle", "#8aa9ff", false], ["threshold", "#ffd07b", true]]) {
    make("polyline", { points: rows.map((row, index) => `${x(index)},${y(row[field])}`).join(" "),
      fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dashed ? "5 5" : "none" });
  }
  rows.forEach((row, index) => {
    const point = make("circle", { cx: x(index), cy: y(row.rmsle), r: 4,
      fill: row.status === "ok" ? "#8aa9ff" : "#ff969f" });
    const title = document.createElementNS(svg.namespaceURI, "title");
    title.textContent = `묶음 ${row.block_id} · v${row.model_version} · ${modeName(row.mode)} · RMSLE ${score(row.rmsle)} · 임계값 ${score(row.threshold)}`;
    point.append(title);
    if (index % Math.max(1, Math.ceil(rows.length / 10)) === 0 || index === rows.length - 1) {
      make("text", { x: x(index), y: 225, "text-anchor": "middle" }, `#${row.block_id}`);
    }
  });
}

function renderScenario(report) {
  const events = report.events || [];
  const retraining = events.findLast(event => event.retraining)?.retraining;
  $("scenario-result").hidden = false;
  $("result-title").textContent = report.scenario === "normal" ? "정상 시나리오 결과" : "드리프트 시나리오 결과";
  let summary = `${events.length}개 묶음을 평가했습니다.`;
  if (retraining?.status === "completed") {
    summary += retraining.promoted ? ` 검증 성능이 개선되어 v${report.active_model_version}로 교체했습니다.` : " 후보가 교체 기준을 충족하지 못해 기존 모델을 유지했습니다.";
  } else if (retraining) {
    const labels = { deferred: "재학습 보류", failed: "재학습 오류", manual_review: "수동 검토 대기", skipped_old_version: "이전 버전의 재학습 생략" };
    summary += ` ${labels[retraining.status] || retraining.status}. ${retraining.reason || ""}`;
  } else if (events.every(event => event.status === "ok")) summary += " 모두 임계값 이내이며 재학습 없이 모델을 유지했습니다.";
  else summary += " 묶음별 판정을 확인하세요.";
  if (report.baseline_required) summary += " 새 기준값 설정이 필요합니다. 시연을 반복하려면 서버를 재시작하세요.";
  $("result-summary").textContent = summary;
  $("previous-score").textContent = score(retraining?.current_rmsle);
  $("candidate-score").textContent = score(retraining?.rmsle);
  $("promotion-result").textContent = retraining?.promoted ? `v${report.active_model_version} 선택` : `v${report.active_model_version} 유지`;
  $("scenario-events").replaceChildren(...events.map(event => element("li",
    `${event.window}번째 묶음 · RMSLE ${score(event.rmsle)} / 임계값 ${score(event.threshold)} · ${statusNames[event.status] || event.status}`)));
  $("scenario-json").textContent = JSON.stringify(report, null, 2);
}

async function runScenario(name) {
  await action(name === "normal" ? "정상 시나리오 실행 중" : "드리프트 감지·재학습 실행 중", async () => {
    notice("서버에서 시나리오를 실행하고 있습니다. 완료될 때까지 기다려 주세요.");
    const report = await api.scenario(name);
    state.completed.add(name);
    $("history-mode").value = "simulation";
    $("history-version").value = "";
    state.predictionPage = state.driftPage = 1;
    renderScenario(report);
    await refreshOverview();
    const retraining = report.events.findLast(event => event.retraining)?.retraining;
    const failed = retraining && ["failed", "deferred", "manual_review"].includes(retraining.status);
    notice(failed ? "평가는 완료됐지만 재학습 상태를 확인해야 합니다. 실행 결과와 로그를 확인하세요."
      : "시나리오가 완료됐습니다. 아래에서 평가 결과와 예측 이력을 확인하세요.", failed ? "warn" : "ok");
  });
}

async function loadLog() {
  try { $("log-content").textContent = (await api.log()).content || "아직 기록된 로그가 없습니다."; }
  catch (error) { $("log-content").textContent = `로그 조회: ${error.message}`; }
}
async function loadDataset() {
  const purpose = $("dataset-purpose").value;
  try {
    const data = await api.dataset(purpose);
    if ($("dataset-purpose").value !== purpose) return;
    $("dataset-status").textContent = data.exists ? `최신 파일: ${data.filename} · ${format(data.rows)}개 영상` : "이 용도로 업로드한 파일이 없습니다.";
  } catch (error) { $("dataset-status").textContent = error.message; }
}

$("run-normal").addEventListener("click", () => runScenario("normal"));
function changeView() { state.predictionPage=state.driftPage=1; state.reviewVideo=null; $('video-detail').hidden=true; $('actual-form').hidden=true; renderVideos(); updateControls(); }
$('drift-scope').addEventListener('change', () => { state.driftPage=1; renderRelated(); updateControls(); });
for (const id of ["video-filter", "video-sort"]) $(id).addEventListener("change", changeView);
document.querySelectorAll('[data-video-view]').forEach(button=>button.addEventListener('click',()=>{ $('video-filter').value=button.dataset.videoView; changeView(); }));
$("run-drift").addEventListener("click", () => runScenario("drift"));
$("refresh").addEventListener("click", () => action("상태 새로고침", refreshOverview));
$("refresh-log").addEventListener("click", () => action("로그 조회", loadLog));
$("filters").addEventListener("submit", event => {
  event.preventDefault();
  action("이력 조회", async () => {
    state.predictionPage = state.driftPage = 1;
    await Promise.all([loadHistory("prediction"), loadHistory("drift")]);
  });
});
for (const kind of ["prediction", "drift"]) {
  for (const [direction, step] of [["prev", -1], ["next", 1]]) {
    $(`${kind}-${direction}`).addEventListener("click", () => action("이력 조회", async () => {
      state[`${kind}Page`] += step;
      if (kind === 'prediction') { $('video-detail').hidden=true; renderVideos(); } else renderRelated();
    }));
  }
}
$("actual-cancel").addEventListener("click", () => { $("actual-form").hidden = true; });
$("actual-form").addEventListener("submit", event => {
  event.preventDefault();
  action("실제 조회수 등록", async () => {
    const result = await api.actual({ prediction_id: $("actual-id").value, target_views_day7: Number($("actual-value").value) });
    $("actual-form").hidden = true;
    await refreshOverview();
    notice(`실제값을 저장했습니다. 평가 상태: ${statusNames[result.drift_check.status] || result.drift_check.status}`, "ok");
  });
});
$("prediction-form").addEventListener("submit", event => {
  event.preventDefault();
  action("영상 CSV 예측", async () => {
    const result = await api.predictCsv($("prediction-file").files[0]);
    $("prediction-status").textContent = `${format(result.predictions.length)}개 영상의 예측을 저장했습니다.`;
    $("history-mode").value = "live"; $("history-version").value = "";
    state.predictionPage = state.driftPage = 1;
    await refreshOverview();
    notice("영상 예측이 완료됐습니다. 목록에서 예측값을 확인하고 실제값을 등록할 수 있습니다.", "ok");
  });
});
$("dataset-purpose").addEventListener("change", loadDataset);
$("dataset-form").addEventListener("submit", event => {
  event.preventDefault();
  action("학습 데이터 업로드", async () => {
    const result = await api.uploadDataset($("dataset-purpose").value, $("dataset-file").files[0]);
    await loadDataset();
    notice(`${format(result.rows)}개 영상의 데이터를 업로드했습니다.`, "ok");
  });
});
$("training-mode").addEventListener("change", () => {
  $("training-epochs").value = $("training-mode").value === "fine_tune" ? "10" : "100";
});
$("training-form").addEventListener("submit", event => {
  event.preventDefault();
  action("모델 학습", async () => {
    const result = await api.train({ mode: $("training-mode").value, epochs: Number($("training-epochs").value), patience: 10 });
    $("training-status").textContent = `검증 RMSLE ${score(result.rmsle)} · ${result.promoted ? `v${result.version} 선택` : "기존 모델 유지"}`;
    await refreshOverview();
    notice("학습·평가를 완료했습니다. 모델 선택 결과와 기준값 상태를 확인하세요.", "ok");
  });
});

await action("대시보드 불러오는 중", async () => {
  try { const response = await fetch('/video-metadata.json'); if(response.ok) displayMetadata=await response.json(); } catch { /* Optional display-only metadata; prediction APIs remain usable. */ }
  await refreshOverview();
  if (state.health) notice("서버의 실제 상태와 저장된 이력을 불러왔습니다.", "ok");
});
