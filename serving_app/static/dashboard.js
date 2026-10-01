import { api } from "./api.js";

const $ = id => document.getElementById(id);
const state = {
  health: null, initialized: false, busy: false, completed: new Set(),
  predictionPage: 1, driftPage: 1, predictionTotal: 0, driftTotal: 0,
};
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
      $("history-mode").value = health.demo_mode ? "simulation" : "live";
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
  await Promise.all([loadHistory("prediction"), loadHistory("drift"), loadLog(), loadDataset()]);
  // 서버가 시연 기록을 비운 경우 이전 페이지의 실행 결과도 제거한다.
  if (state.health?.demo_mode && filters().mode === "simulation" && !filters().model_version &&
      state.predictionTotal === 0 && state.driftTotal === 0) {
    state.completed.clear();
    $("scenario-result").hidden = true;
  }
  updateControls();
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
  const params = { ...filters(), page: state[`${kind}Page`], page_size: pageSizes[kind] };
  const errorBox = $(`${kind}-error`);
  errorBox.textContent = "";
  try {
    let data = await (kind === "prediction" ? api.predictions(params) : api.drift(params));
    // 서버 재시작으로 이력이 줄어들면 존재하는 페이지로 돌아간다.
    if (params.page > 1 && data.items.length === 0) {
      state[`${kind}Page`] = Math.max(1, Math.ceil(data.total / params.page_size));
      return loadHistory(kind);
    }
    state[`${kind}Total`] = data.total;
    $(`${kind}-page-label`).textContent = `${format(data.total)}건 · ${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))}페이지`;
    if (kind === "prediction") {
      renderTable("prediction-rows", data.items, [
        row => row.video_id, row => format(row.predicted_views_day7),
        row => row.target_views_day7 == null ? "미등록" : format(row.target_views_day7),
        row => `v${row.model_version}`, row => modeName(row.mode),
        row => new Date(row.predicted_at).toLocaleString("ko-KR"), actualButton,
      ], "예측 이력이 없습니다. 시나리오를 실행하거나 영상 CSV로 예측해 보세요.");
    } else {
      renderTable("drift-rows", data.items, [
        row => `#${row.block_id}`, row => score(row.rmsle), row => score(row.threshold),
        row => badge(row.status), row => `${row.consecutive_exceeds}회`,
        row => `v${row.model_version}`, row => modeName(row.mode),
      ], "완료된 평가 묶음이 없습니다. 실제값 30개가 모이면 평가합니다.");
      renderChart(data.items);
    }
  } catch (error) {
    errorBox.textContent = `이력 조회 실패: ${error.message}`;
    $(`${kind}-rows`).replaceChildren();
    $(`${kind}-page-label`).textContent = "불러오기 실패";
    if (kind === "drift") renderChart([]);
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
      await loadHistory(kind);
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
  await refreshOverview();
  if (state.health) notice("서버의 실제 상태와 저장된 이력을 불러왔습니다.", "ok");
});
