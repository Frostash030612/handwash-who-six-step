'use strict';
const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('canvas');
const COLORS = ['var(--other)', 'var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)'];
const NAMES = ['Other', 'Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'];
const MODEL_NAMES = {yolo: 'YOLO', yolo_gru: 'YOLO + GRU', boundary_gru: 'YOLO + Adaptive Boundary-Aware GRU', videomae: 'VideoMAE'};
const STATUS_TEXT = {
  '选择模型并开启摄像头后开始。': 'Choose a model and enable the camera preview to begin.',
  '模型已就绪，正在采集。': 'Model ready. Capturing frames.',
  '正在识别，洗手计时为实时估计。': 'Recognizing actions. Washing time is a live estimate.',
  '正在汇总已缓存的预测…': 'Preparing the report from cached predictions…',
  '识别完成，洗手时长和动作顺序见下方。': 'Session complete. Review the washing time and step order below.',
  '正在加载所选模型…': 'Loading the selected model…',
  '正在处理最后几帧…': 'Processing the last received frames…',
  'VideoMAE 正在收集首个 16 帧窗口（约 3 秒）。': 'VideoMAE is buffering its first 16-frame window (about 3 seconds).',
  '无效图像': 'The received image is invalid.',
  'VideoMAE 权重或预处理配置尚未完整就绪': 'VideoMAE weights or preprocessing files are not ready.',
  'VideoMAE 窗口或类别数量与自采模型不一致': 'The VideoMAE frame window or class count does not match this model.',
  'VideoMAE 类别映射不一致': 'The VideoMAE class mapping does not match.',
  'YOLO 与时序模型权重不匹配': 'The YOLO and temporal model weights do not match.',
  '本次没有收到画面，请重新开始': 'No frames were received. Please start a new session.',
  '录像时长不足': 'The recording is too short to analyze.',
  '尚有帧未完成推理，不能生成完整结果': 'Some received frames are still being processed.',
  '请求过大': 'The request is too large.',
  '数据接收不完整': 'The uploaded frame was incomplete.',
  '请先选择一个模型，再开始': 'Select a model before starting.',
  '所选模型权重尚未完整就绪': 'The selected model weights are not ready.',
  '上一段识别还未结束': 'The previous session has not finished.',
  '图像尺寸过大': 'The image dimensions are too large.',
  '当前没有对应的录像会话': 'No matching recording session is active.',
  '超出测试时长': 'The recording exceeds the time limit.',
  '采样时间未递增': 'The frame timestamp did not advance.',
  '无效结束时间': 'The recording end time is invalid.',
  '未能读取摄像头画面': 'Could not capture a camera frame.',
  '请先开启摄像头': 'Enable the camera preview first.',
  '录像会话已改变': 'The recording session changed.',
  '服务无响应': 'The local service did not respond.',
  '本机服务需要更新，请等待重启。': 'The local service needs to be restarted.'
};
const english = value => STATUS_TEXT[value] || value;
function englishReason(value) {
  let match = value.match(/^洗手总时长不足：([\d.]+) 秒，要求至少 ([\d.]+) 秒$/);
  if (match) return `Recognized washing time is ${match[1]} sec; at least ${match[2]} sec is required.`;
  match = value.match(/^未识别到动作 (.*)$/);
  if (match) return `Steps not detected: ${match[1].replaceAll('、', ', ')}.`;
  if (value === '动作顺序不符合 1 → 2 → 3 → 4 → 5 → 6') return 'The detected steps are out of order.';
  match = value.match(/^已识别洗手 ([\d.]+) 秒，另有 ([\d.]+) 秒未得到预测，暂不能确认时长达标$/);
  if (match) return `${match[1]} sec of washing recognized; another ${match[2]} sec was unobserved, so the time requirement is uncertain.`;
  return english(value);
}
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
  if (!response.ok) throw Error(data.error || 'Request failed');
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
    ? 'Top score in our internal matched-frame comparison (632 test points; 99.53% accuracy). Its 16-frame window needs ~3 s for the first prediction and uses ~1.4 s of future frames, so feedback trails the live action. Best as an offline benchmark, not an immediate camera coach.'
    : 'Only this model runs. Selection locks when the session starts.';
  $('report').hidden = true;
  renderedSession = null;
  resetCurrent();
  showWash(0, false);
  controls();
}
function resetCurrent() {
  for (const step of document.querySelectorAll('[data-step]')) step.classList.remove('active');
  $('currentNo').textContent = '–'; $('currentName').textContent = 'Waiting to start';
  $('confidenceText').textContent = 'Confidence —'; $('confidenceBar').style.width = '0%';
  $('overlay').hidden = true;
}
function current(label, confidence) {
  const k = Number(label), p = Number(confidence);
  if (!Number.isInteger(k) || k < 0 || k > 6 || !Number.isFinite(p)) return;
  for (const step of document.querySelectorAll('[data-step]')) step.classList.toggle('active', Number(step.dataset.step) === k);
  $('currentNo').style.setProperty('--c', COLORS[k]);
  for (const id of ['currentNo', 'liveNo']) { $(id).textContent = k || '·'; $(id).style.setProperty('--c', COLORS[k]); }
  $('currentName').textContent = $('liveName').textContent = NAMES[k];
  $('confidenceText').textContent = `Confidence ${(p * 100).toFixed(1)}%`;
  $('liveConf').textContent = `${(p * 100).toFixed(1)}%`;
  $('confidenceBar').style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
  $('overlay').hidden = !stream?.active;
}
function showWash(value, final) {
  $('washLabel').firstChild.textContent = 'Total washing time ';
  $('washLabel').querySelector('.live-estimate').hidden = final;
  $('washSeconds').textContent = seconds(value);
  $('minimumSeconds').textContent = minimumSeconds;
  $('washProgress').style.width = `${Math.min(100, value / minimumSeconds * 100)}%`;
  const met = value + 1e-9 >= minimumSeconds;
  $('washHint').textContent = met ? (final ? 'Time requirement met. Check the step sequence for the full result.' : 'Live estimate meets the time requirement. Confirm after finishing.')
    : `${seconds(minimumSeconds - value)} sec remaining. Other and unobserved time are excluded.`;
  $('washHint').className = 'message ' + (met ? 'ok' : '');
}
function renderResult(result) {
  if (result.model !== selected() || result.schema_version !== 2) return;
  if (renderedSession === result.session) return;
  renderedSession = result.session;
  const a = result.assessment, prediction = result.prediction;
  $('report').hidden = false;
  $('reportTitle').textContent = `${MODEL_NAMES[result.model]} · Session results`;
  $('recordingInfo').textContent = `${seconds(result.recorded_seconds)} sec · ${result.captured_frames} frames received`;
  $('verdict').textContent = ({'达标': 'Complete', '未达标': 'Incomplete', '待确认': 'Needs review'})[a.verdict] || a.verdict;
  $('verdict').className = 'verdict ' + (a.passed ? 'pass' : a.verdict === '待确认' ? 'pending' : 'fail');
  $('finalWash').textContent = `${seconds(a.wash_seconds)} sec · ${a.duration_met ? 'met' : a.duration_status === 'insufficient_evidence' ? 'uncertain' : 'short'}`;
  $('finalWash').className = a.duration_met ? 'pass' : 'fail';
  $('finalSteps').textContent = `${6 - a.missing_steps.length} / 6`;
  $('finalOrder').textContent = a.order_correct ? 'Correct' : !a.steps_complete ? 'Incomplete' : 'Out of order';
  $('reasons').replaceChildren(...a.reasons.map(text => { const li = document.createElement('li'); li.textContent = englishReason(text); return li; }));
  $('order').textContent = prediction.order.length ? prediction.order.map(step => `Step ${step}`).join(' → ') : 'No confirmed washing steps';
  $('timeline').replaceChildren();
  for (const segment of prediction.segments) {
    const node = document.createElement('button');
    const name = segment.class < 0 ? 'Unobserved' : NAMES[segment.class];
    node.className = 'segment';
    node.style.setProperty('--c', segment.class < 0 ? 'var(--border)' : COLORS[segment.class]);
    node.style.width = `${segment.duration_s / result.recorded_seconds * 100}%`;
    node.title = `${seconds(segment.start_s)}–${seconds(segment.end_s)} sec: ${name}${segment.status === 'unconfirmed' ? ' (not confirmed in the step sequence)' : ''}`;
    node.setAttribute('aria-label', node.title);
    $('timeline').append(node);
  }
  $('coverage').textContent = `Washing ${seconds(a.wash_seconds)} sec · Other ${seconds(a.background_seconds)} sec · Unobserved ${seconds(a.unknown_seconds)} sec`;
  $('durationDetails').textContent = `Recording: ${seconds(result.recorded_seconds)} sec. Recognized washing: ${seconds(a.wash_seconds)} sec. Other: ${seconds(a.background_seconds)} sec. Unobserved: ${seconds(a.unknown_seconds)} sec. No individual-step time requirement.`;
  $('computeDetails').textContent = `Inference during recording: ${seconds(prediction.total_compute_s)} sec. Final summary: ${milliseconds(result.finalization_ms)} (not including queue wait).`;
  showWash(a.wash_seconds, true);
}
function updateState(data) {
  state = data; connected = data.api_version === 2;
  if (!connected) { say('This page requires an updated local service. Please restart it.', 'error'); controls(); return; }
  minimumSeconds = data.minimum_wash_seconds;
  for (const label of document.querySelectorAll('[data-minimum]')) {
    const text = String(minimumSeconds);
    if (label.textContent !== text) label.textContent = text;
  }
  available = Object.fromEntries(data.available_models.map(m => [m.id, m.available]));
  for (const option of $('modelSelect').options) {
    const disabled = !available[option.value];
    const label = MODEL_NAMES[option.value] + (disabled ? ' (weights unavailable)' : '');
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
  if (data.phase === 'error') { pill('Error'); say(english(data.message), 'error'); }
  else if (data.phase === 'preparing') { pill('Loading model', 'ready'); say(english(data.message)); }
  else if (data.phase === 'recording') { pill('Live', 'running'); say(!recording && !starting ? 'A previous session is still running. Finish it to save the frames already received.' : english(data.message)); }
  else if (['flushing', 'analyzing'].includes(data.phase)) { pill('Finalizing', 'ready'); say(english(data.message)); }
  else if (data.phase === 'done' && matching) { pill('Complete', 'done'); say('Session complete. Select a model to begin another.', 'ok'); }
  else { pill('Ready', 'ready'); if (!starting) say(stream?.active ? 'Camera connected. Start when ready.' : 'Choose a model and enable the camera preview to begin.'); }
  controls();
}
async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try { const response = await fetch('/status', {cache: 'no-store'}); if (!response.ok) throw Error('服务无响应'); updateState(await response.json()); }
  catch (error) { connected = false; pill('Disconnected'); say(`Cannot reach the local service: ${error.message}`, 'error'); controls(); }
  finally { pollBusy = false; }
}
async function listCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) { say('Camera access is unavailable. Open this page in Google Chrome to use the camera.', 'error'); return; }
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const chosen = stream?.getVideoTracks()[0]?.getSettings().deviceId || $('cameras').value;
  $('cameras').replaceChildren(...devices.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
  if (!devices.length) $('cameras').append(new Option('Default camera', ''));
  if (devices.some(d => d.deviceId === chosen)) $('cameras').value = chosen;
}
async function openCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('Open this local address in Google Chrome to use the camera.');
    if (stream) stream.getTracks().forEach(t => t.stop());
    const id = $('cameras').value;
    stream = await navigator.mediaDevices.getUserMedia({video: {...(id ? {deviceId: {exact: id}} : {}), width: {ideal: 1920}, height: {ideal: 1080}}, audio: false});
    video.srcObject = stream;
    await video.play();
    $('empty').hidden = true;
    await listCameras();
    stream.getVideoTracks()[0].addEventListener('ended', () => { if (recording) stopSession(); controls(); });
    say('Camera connected.', 'ok');
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
  catch (error) { say(`Capture failed: ${english(error.message)}`, 'error'); }
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
  } catch (error) { say(english(error.message), 'error'); session = null; }
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
  } catch (error) { say(`Could not finish: ${english(error.message)}`, 'error'); recording = state.phase === 'recording'; }
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
  label.append(dot, document.createTextNode(i < 0 ? 'Unobserved' : NAMES[i])); $('legend').append(label);
}
setInterval(() => {
  if (!recording) return;
  const elapsed = (performance.now() - startedAt) / 1000;
  $('clock').textContent = `${seconds(elapsed)} sec`;
  $('elapsed').textContent = `Recording ${seconds(elapsed)} sec`;
}, 100);
selectionChanged(); listCameras().catch(() => {}); void poll(); setInterval(poll, 500);
