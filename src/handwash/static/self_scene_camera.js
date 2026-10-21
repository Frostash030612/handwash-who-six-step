'use strict';
const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('canvas');
const COLORS = ['var(--other)', 'var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)'];
const NAMES = ['其他', '动作 1', '动作 2', '动作 3', '动作 4', '动作 5', '动作 6'];
const MODEL_NAMES = {yolo: 'YOLO', yolo_gru: 'YOLO + GRU', boundary_gru: 'YOLO + 自适应边界感知 GRU', videomae: 'VideoMAE'};
let stream = null, session = null, timer = null, pending = null, startedAt = 0;
let recording = false, starting = false, stopping = false, pollBusy = false, connected = false;
let state = {}, minimumSeconds = 40, available = {}, renderedSession = null;
const activePhases = ['preparing', 'recording', 'flushing', 'analyzing'];
const milliseconds = x => Number.isFinite(x) ? `${Math.round(x)} ms` : '—';
const seconds = x => Number(x || 0).toFixed(1);
const selected = () => $('modelSelect').value;
function say(text, kind = '') { $('message').textContent = text; $('message').className = 'message ' + kind; }
function pill(text, kind = '') { $('pill').textContent = text; $('pill').className = 'pill ' + kind; }
async function post(path, body, isJson = false) {
  const response = await fetch(path, {method: 'POST', headers: isJson ? {'Content-Type': 'application/json'} : {}, body: isJson ? JSON.stringify(body) : body});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '请求失败');
  return data;
}
function controls() {
  const locked = starting || recording || stopping || activePhases.includes(state.phase);
  // Reassigning native select state while its menu is open can redraw the popup.
  if ($('modelSelect').disabled !== locked) $('modelSelect').disabled = locked;
  if ($('cameras').disabled !== locked) $('cameras').disabled = locked;
  $('preview').disabled = locked;
  $('close').disabled = locked || !stream?.active;
  $('start').disabled = locked || !connected || !stream?.active || !available[selected()];
  $('stop').disabled = starting || stopping || (!recording && state.phase !== 'recording');
}
function selectionChanged() {
  $('modelBadge').textContent = MODEL_NAMES[selected()];
  $('modelHint').textContent = selected() === 'videomae'
    ? 'VideoMAE 约 3 秒后给出首个结果，再每 0.8 秒更新。窗口标签相对画面约延迟 1.4 秒，未预测时间不计入洗手时长。'
    : '一次只使用所选模型，开始后锁定选择。';
  $('report').hidden = true;
  renderedSession = null;
  resetCurrent();
  showWash(0, false);
  controls();
}
function resetCurrent() {
  for (const step of document.querySelectorAll('[data-step]')) step.classList.remove('active');
  $('currentNo').textContent = '–'; $('currentName').textContent = '等待开始';
  $('confidenceText').textContent = '置信度 —'; $('confidenceBar').style.width = '0%';
  $('overlay').hidden = true;
}
function current(label, confidence) {
  const k = Number(label), p = Number(confidence);
  if (!Number.isInteger(k) || k < 0 || k > 6 || !Number.isFinite(p)) return;
  for (const step of document.querySelectorAll('[data-step]')) step.classList.toggle('active', Number(step.dataset.step) === k);
  $('currentNo').style.setProperty('--c', COLORS[k]);
  for (const id of ['currentNo', 'liveNo']) { $(id).textContent = k || '·'; $(id).style.setProperty('--c', COLORS[k]); }
  $('currentName').textContent = $('liveName').textContent = NAMES[k];
  $('confidenceText').textContent = `置信度 ${(p * 100).toFixed(1)}%`;
  $('liveConf').textContent = `${(p * 100).toFixed(1)}%`;
  $('confidenceBar').style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
  $('overlay').hidden = !stream?.active;
}
function showWash(value, final) {
  $('washLabel').textContent = final ? '本次累计洗手时长' : '累计洗手时长（实时估计）';
  $('washSeconds').textContent = seconds(value);
  $('minimumSeconds').textContent = minimumSeconds;
  $('washProgress').style.width = `${Math.min(100, value / minimumSeconds * 100)}%`;
  const met = value + 1e-9 >= minimumSeconds;
  $('washHint').textContent = met ? (final ? '总时长已达标；完整结论还需结合动作和顺序。' : '实时估计已达到时长要求，结束后确认。')
    : `距离时长要求还差 ${seconds(minimumSeconds - value)} 秒；“其他”和未识别时间不计入。`;
  $('washHint').className = 'message ' + (met ? 'ok' : '');
}
function renderResult(result) {
  if (result.model !== selected() || result.schema_version !== 2) return;
  if (renderedSession === result.session) return;
  renderedSession = result.session;
  const a = result.assessment, prediction = result.prediction;
  $('report').hidden = false;
  $('reportTitle').textContent = `${result.model_name} · 本次结果`;
  $('recordingInfo').textContent = `录像 ${seconds(result.recorded_seconds)} 秒 · ${result.captured_frames} 帧`;
  $('verdict').textContent = a.verdict;
  $('verdict').className = 'verdict ' + (a.passed ? 'pass' : a.verdict === '待确认' ? 'pending' : 'fail');
  $('finalWash').textContent = `${seconds(a.wash_seconds)} 秒 · ${a.duration_met ? '达标' : a.duration_status === 'insufficient_evidence' ? '待确认' : '不足'}`;
  $('finalWash').className = a.duration_met ? 'pass' : 'fail';
  $('finalSteps').textContent = `${6 - a.missing_steps.length} / 6`;
  $('finalOrder').textContent = a.order_correct ? '正确' : !a.steps_complete ? '动作不完整' : '顺序不符';
  $('reasons').replaceChildren(...a.reasons.map(text => { const li = document.createElement('li'); li.textContent = text; return li; }));
  $('order').textContent = prediction.order.length ? prediction.order.join(' → ') : '暂无可确认的洗手动作';
  $('timeline').replaceChildren();
  for (const segment of prediction.segments) {
    const node = document.createElement('button');
    const name = segment.class < 0 ? '未识别' : NAMES[segment.class];
    node.className = 'segment';
    node.style.setProperty('--c', segment.class < 0 ? 'var(--border)' : COLORS[segment.class]);
    node.style.width = `${segment.duration_s / result.recorded_seconds * 100}%`;
    node.title = `${seconds(segment.start_s)}–${seconds(segment.end_s)} 秒：${name}${segment.status === 'unconfirmed' ? '（未进入动作链）' : ''}`;
    node.setAttribute('aria-label', node.title);
    $('timeline').append(node);
  }
  $('coverage').textContent = `洗手 ${seconds(a.wash_seconds)} 秒 · 其他 ${seconds(a.background_seconds)} 秒 · 未识别 ${seconds(a.unknown_seconds)} 秒`;
  $('durationDetails').textContent = `录像总时长 ${seconds(result.recorded_seconds)} 秒；累计洗手 ${seconds(a.wash_seconds)} 秒；其他 ${seconds(a.background_seconds)} 秒；未识别 ${seconds(a.unknown_seconds)} 秒。各动作不单独考核时长。`;
  $('computeDetails').textContent = `录制期间推理累计 ${seconds(prediction.total_compute_s)} 秒；结束后汇总 ${milliseconds(result.finalization_ms)}（不含队列等待）。`;
  showWash(a.wash_seconds, true);
}
function updateState(data) {
  state = data; connected = data.api_version === 2;
  if (!connected) { say('服务需要更新，请等待本机服务重启。', 'error'); controls(); return; }
  minimumSeconds = data.minimum_wash_seconds;
  for (const label of document.querySelectorAll('[data-minimum]')) {
    const text = String(minimumSeconds);
    if (label.textContent !== text) label.textContent = text;
  }
  available = Object.fromEntries(data.available_models.map(m => [m.id, m.available]));
  for (const option of $('modelSelect').options) {
    const disabled = !available[option.value];
    const label = MODEL_NAMES[option.value] + (disabled ? '（权重尚未就绪）' : '');
    // Status is polled every 500 ms. Preserve option nodes/text when unchanged,
    // otherwise an open native menu flickers even though its labels are identical.
    if (option.disabled !== disabled) option.disabled = disabled;
    if (option.textContent !== label) option.textContent = label;
  }
  const matching = data.model === selected();
  if (matching) {
    if (data.label != null) current(data.label, data.confidence);
    showWash(data.live_wash_seconds || 0, data.phase === 'done');
    $('device').textContent = data.device;
    $('frames').textContent = data.frames || 0;
    $('infer').textContent = milliseconds(data.inference_ms);
    $('return').textContent = milliseconds(data.capture_to_result_ms);
  }
  if (data.phase === 'done' && matching && data.result) renderResult(data.result);
  if (data.phase === 'error') { pill('识别失败'); say(data.message, 'error'); }
  else if (data.phase === 'preparing') { pill('加载模型', 'ready'); say(data.message); }
  else if (data.phase === 'recording') { pill('识别中', 'running'); say(!recording && !starting ? '上一会话仍在运行，请点击结束保存已收到的画面。' : data.message); }
  else if (['flushing', 'analyzing'].includes(data.phase)) { pill('整理结果', 'ready'); say(data.message); }
  else if (data.phase === 'done' && matching) { pill('完成', 'done'); say('本次识别完成。可以选择模型并开始下一次。', 'ok'); }
  else { pill('就绪', 'ready'); if (!starting) say(stream?.active ? '摄像头已连接，准备好后开始。' : '选择模型，开启摄像头预览后开始。'); }
  controls();
}
async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try { const response = await fetch('/status', {cache: 'no-store'}); if (!response.ok) throw Error('服务无响应'); updateState(await response.json()); }
  catch (error) { connected = false; pill('连接失败'); say(`无法连接本机服务：${error.message}`, 'error'); controls(); }
  finally { pollBusy = false; }
}
async function listCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) { say('当前浏览器没有开放摄像头，请在 Google Chrome 中打开此地址。', 'error'); return; }
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const chosen = stream?.getVideoTracks()[0]?.getSettings().deviceId || $('cameras').value;
  $('cameras').replaceChildren(...devices.map((d, i) => new Option(d.label || `摄像头 ${i + 1}`, d.deviceId)));
  if (!devices.length) $('cameras').append(new Option('默认摄像头', ''));
  if (devices.some(d => d.deviceId === chosen)) $('cameras').value = chosen;
}
async function openCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('请在 Google Chrome 中打开本机地址以使用摄像头。');
    if (stream) stream.getTracks().forEach(t => t.stop());
    const id = $('cameras').value;
    stream = await navigator.mediaDevices.getUserMedia({video: {...(id ? {deviceId: {exact: id}} : {}), width: {ideal: 1920}, height: {ideal: 1080}}, audio: false});
    video.srcObject = stream;
    await video.play();
    $('empty').hidden = true;
    await listCameras();
    stream.getVideoTracks()[0].addEventListener('ended', () => { if (recording) stopSession(); controls(); });
    say('摄像头已连接。', 'ok');
  } catch (error) { stream = null; $('empty').hidden = false; say(error.message, 'error'); }
  controls();
}
async function sendFrame() {
  if (!recording || pending || !video.videoWidth) return;
  const time = (performance.now() - startedAt) / 1000;
  if (time >= 120) { void stopSession(); return; }
  pending = (async () => {
    const scale = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .92));
    if (!blob) throw Error('未能读取摄像头画面');
    await post(`/frame?session=${encodeURIComponent(session)}&t=${time.toFixed(5)}`, blob);
  })();
  try { await pending; }
  catch (error) { say(`采集失败：${error.message}`, 'error'); }
  finally { pending = null; }
}
async function startSession() {
  if (starting || recording) return;
  starting = true; controls();
  try {
    if (!stream?.active) throw Error('请先开启摄像头');
    const response = await post('/start', {model: selected()}, true);
    session = response.session;
    $('report').hidden = true; renderedSession = null; resetCurrent(); showWash(0, false);
    // Wait for only the chosen model to load before the recording clock starts.
    for (;;) {
      const statusResponse = await fetch('/status', {cache: 'no-store'});
      const data = await statusResponse.json(); updateState(data);
      if (data.session !== session) throw Error('录像会话已改变');
      if (data.phase === 'error') throw Error(data.message);
      if (data.phase === 'recording') break;
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    recording = true; startedAt = performance.now(); $('liveChip').hidden = false;
    await sendFrame(); timer = setInterval(() => void sendFrame(), 200);
  } catch (error) { say(error.message, 'error'); session = null; }
  finally { starting = false; controls(); }
}
async function stopSession() {
  if (stopping || starting) return;
  const recovering = !session && state.phase === 'recording';
  if (recovering) session = state.session;
  if (!session || (!recording && !recovering)) return;
  stopping = true; recording = false; clearInterval(timer); timer = null;
  const elapsed = recovering ? (state.elapsed_s || 0) : Math.min(125, (performance.now() - startedAt) / 1000);
  $('liveChip').hidden = true; controls();
  try {
    if (pending) await pending.catch(() => {});
    await post('/stop', {session, elapsed_s: elapsed}, true);
    session = null;
    await poll();
  } catch (error) { say(`结束失败：${error.message}`, 'error'); recording = state.phase === 'recording'; }
  finally { stopping = false; controls(); }
}
$('preview').addEventListener('click', openCamera);
$('cameras').addEventListener('change', openCamera);
$('modelSelect').addEventListener('change', selectionChanged);
$('start').addEventListener('click', startSession);
$('stop').addEventListener('click', stopSession);
$('close').addEventListener('click', () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; video.srcObject = null; $('empty').hidden = false; $('overlay').hidden = true; controls();
});
for (let i = -1; i < NAMES.length; i++) {
  const label = document.createElement('span'), dot = document.createElement('i');
  dot.className = 'dot'; dot.style.setProperty('--c', i < 0 ? 'var(--border)' : COLORS[i]);
  label.append(dot, document.createTextNode(i < 0 ? '未识别' : NAMES[i])); $('legend').append(label);
}
setInterval(() => {
  if (!recording) return;
  const elapsed = (performance.now() - startedAt) / 1000;
  $('clock').textContent = `${seconds(elapsed)} 秒`;
  $('elapsed').textContent = `录像 ${seconds(elapsed)} 秒`;
}, 100);
selectionChanged(); listCameras().catch(() => {}); void poll(); setInterval(poll, 500);
