/* ============================================================
 * 医疗健康助手 · 前端逻辑
 * 认证 / 会话 / SSE 流式对话 / XSS 安全 markdown
 * ============================================================ */

const API_BASE =
  new URLSearchParams(location.search).get('api') ||
  (localStorage.getItem('apiBase') || 'http://localhost:8000');

const LS_TOKEN = 'accessToken';
const LS_USER = 'currentUser';

const state = {
  token: null,
  userId: null,
  sessionId: null,
  busy: false,
  turns: 0,
};

const $ = (id) => document.getElementById(id);
const show = (el) => { el.hidden = false; };
const hide = (el) => { el.hidden = true; };

const AI_SVG = '<svg viewBox="0 0 40 40" width="30" height="30"><rect x="2" y="2" width="36" height="36" rx="11" fill="var(--c-accent)"/><path d="M20 12v16M12 20h16" stroke="var(--c-accent-ink)" stroke-width="4" stroke-linecap="round"/></svg>';

function nowTime() {
  const d = new Date();
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
}

/* ---------- XSS 安全 markdown ---------- */
function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function inlineMd(s) {
  s = s.replace(/`([^`]+)`/g, (m, p) => '<code>' + p + '</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, (m, p) => '<strong>' + p + '</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, (m, a, p) => a + '<em>' + p + '</em>');
  return s;
}

function renderMarkdown(src) {
  if (!src) return '';
  const text = escapeHtml(String(src).replace(/\r\n/g, '\n').replace(/\r/g, '\n'));
  const lines = text.split('\n');
  const out = [];
  let listType = null;
  const closeList = () => { if (listType) { out.push('</' + listType + '>'); listType = null; } };

  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    const h3 = line.match(/^###\s+(.+)$/);
    const h2 = line.match(/^##\s+(.+)$/);
    const h1 = line.match(/^#\s+(.+)$/);
    if (h1 || h2 || h3) {
      closeList();
      const lvl = h1 ? 2 : h2 ? 3 : 4;
      out.push('<h' + lvl + '>' + inlineMd((h1 || h2 || h3)[1]) + '</h' + lvl + '>');
      i++; continue;
    }
    const ul = line.match(/^[-*]\s+(.+)$/);
    const ol = line.match(/^\d+\.\s+(.+)$/);
    if (ul || ol) {
      const tag = ul ? 'ul' : 'ol';
      if (listType !== tag) { closeList(); out.push('<' + tag + '>'); listType = tag; }
      out.push('<li>' + inlineMd((ul || ol)[1]) + '</li>');
      i++; continue;
    }
    if (/^\s*(?:-{3,}|\*{3,})\s*$/.test(line)) { closeList(); out.push('<hr>'); i++; continue; }
    closeList();
    const para = [];
    while (i < lines.length) {
      const l = lines[i];
      if (!l.trim() || /^(?:#\s|[-*]\s|\d+\.\s|-{3,}\s*$)/.test(l)) break;
      para.push(inlineMd(l));
      i++;
    }
    if (para.length) out.push('<p>' + para.join('<br>') + '</p>');
    else i++;
  }
  closeList();
  return out.join('');
}

/* ---------- 消息渲染 ---------- */
const INTENT_LABELS = {
  drug_conflict: '药物冲突', drug_record: '用药记录', drug_query: '用药咨询',
  lab_report: '化验解读', archive: '档案查询', general: '通用问答',
  multi: '多意图', drug: '药物', lab: '化验',
};

function scrollToBottom() {
  const sc = $('chatScroll');
  sc.scrollTop = sc.scrollHeight;
}

function addMessage(role, content, meta) {
  const list = $('messageList');
  hide($('emptyState'));

  const el = document.createElement('article');
  el.className = 'msg msg--' + role;

  const body = document.createElement('div');
  body.className = 'msg-body';

  const metaEl = document.createElement('div');
  metaEl.className = 'msg-meta';

  if (role === 'assistant') {
    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.innerHTML = AI_SVG;
    el.appendChild(av);

    const author = document.createElement('span');
    author.className = 'msg-author';
    author.textContent = '医疗助手';
    metaEl.appendChild(author);
    if (meta && meta.intent) {
      metaEl.insertAdjacentHTML('beforeend', intentChip(meta));
    }
    const t = document.createElement('time');
    t.textContent = nowTime();
    metaEl.appendChild(t);
    body.appendChild(metaEl);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = content ? renderMarkdown(content) : '<span class="streaming-cursor"></span>';
    body.appendChild(bubble);
    el._bubble = bubble;
  } else {
    const t = document.createElement('time');
    t.textContent = nowTime();
    metaEl.appendChild(t);
    body.appendChild(metaEl);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;
    body.appendChild(bubble);

    const av = document.createElement('div');
    av.className = 'msg-avatar avatar avatar--user';
    av.textContent = '我';
    el.appendChild(body);
    el.appendChild(av);
  }
  if (role === 'assistant') el.appendChild(body);

  list.appendChild(el);
  scrollToBottom();
  return el;
}

function intentChip(meta) {
  if (!meta || !meta.intent) return '';
  const label = INTENT_LABELS[meta.intent] || meta.intent;
  let chip = '<span class="msg-intent">' + escapeHtml(label);
  if (meta.confidence != null) chip += ' · ' + Math.round(meta.confidence * 100) + '%';
  return chip + '</span>';
}

function setIntentChip(el, intent, confidence) {
  if (!el || !intent) return;
  const metaEl = el.querySelector('.msg-meta');
  if (!metaEl || metaEl.querySelector('.msg-intent')) return;
  metaEl.insertAdjacentHTML('beforeend', intentChip({ intent, confidence }));
}

/* ---------- 诊断面板 ---------- */
function updateDiagnostics(d) {
  if (!d) return;
  if (d.conversation_turns != null) { state.turns = d.conversation_turns; $('diagTurns').textContent = d.conversation_turns; }
  if (d.needs_confirmation != null) $('diagConfirm').textContent = d.needs_confirmation ? '是' : '否';
  const ia = d.intent_analysis || {};
  if (ia.intent_type || d.intent) {
    const it = ia.intent_type || d.intent || '';
    $('diagIntent').textContent = INTENT_LABELS[it] || it;
    $('diagTarget').textContent = ia.target_name || d.target_agent || '—';
    $('diagConfidence').textContent = ia.confidence != null ? Math.round(ia.confidence * 100) + '%' : '—';
    $('diagReason').textContent = ia.reason || '—';
  } else if (d.target_agent) {
    $('diagTarget').textContent = d.target_agent;
  }
}

/* ---------- 输入区 ---------- */
function autogrow() {
  const ta = $('userInput');
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 168) + 'px';
}

function setBusy(b) {
  state.busy = b;
  $('sendBtn').disabled = b;
  $('userInput').disabled = b;
  document.querySelectorAll('.cap-card').forEach((c) => { c.disabled = b; });
}

/* ---------- 认证 ---------- */
async function api(path, options) {
  const headers = { 'Content-Type': 'application/json' };
  if (state.token) headers.Authorization = 'Bearer ' + state.token;
  return fetch(API_BASE + path, { ...options, headers });
}

function showAuthError(msg) {
  const el = $('authError');
  el.textContent = msg;
  show(el);
}

function setAuthBusy(b) {
  $('authSubmit').disabled = b;
  $('authSubmit').classList.toggle('is-loading', b);
}

function refreshSessionDisplay() {
  $('sessionId').textContent = state.sessionId ? state.sessionId.slice(0, 8) + '…' : '新会话';
  $('diagSession').textContent = state.sessionId || '—';
}

function enterApp() {
  hide($('authView'));
  show($('appView'));
  $('sideUserName').textContent = state.userId;
  $('sideAvatar').textContent = '我';
  refreshSessionDisplay();
  setTimeout(() => $('userInput').focus(), 50);
}

function leaveApp() {
  state.token = null; state.userId = null; state.sessionId = null;
  state.turns = 0;
  localStorage.removeItem(LS_TOKEN);
  localStorage.removeItem(LS_USER);
  $('messageList').innerHTML = '';
  show($('emptyState'));
  $('diagTurns').textContent = '0';
  $('diagIntent').textContent = '—';
  $('diagTarget').textContent = '—';
  $('diagConfidence').textContent = '—';
  $('diagReason').textContent = '—';
  refreshSessionDisplay();
  hide($('appView'));
  show($('authView'));
}

function startNewChat() {
  state.sessionId = null;
  state.turns = 0;
  $('messageList').innerHTML = '';
  show($('emptyState'));
  $('diagTurns').textContent = '0';
  $('diagIntent').textContent = '—';
  $('diagTarget').textContent = '—';
  $('diagConfidence').textContent = '—';
  $('diagReason').textContent = '—';
  refreshSessionDisplay();
  closeSidebar();
  $('userInput').focus();
}

async function doLogin(phone, password) {
  const res = await api('/api/v1/user/login', {
    method: 'POST',
    body: JSON.stringify({ phone, password }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body.data || !body.data.access_token) {
    throw new Error(body.detail || '登录失败，请检查手机号或密码');
  }
  state.token = body.data.access_token;
  state.userId = body.data.user_id;
  localStorage.setItem(LS_TOKEN, state.token);
  localStorage.setItem(LS_USER, state.userId);
}

async function doRegister(phone, password, nickname) {
  const res = await api('/api/v1/user/register', {
    method: 'POST',
    body: JSON.stringify({ phone, password, user_nickname: nickname || ('用户' + phone) }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body.data || !body.data.user_id) {
    throw new Error(body.detail || '注册失败，请检查输入');
  }
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const mode = document.querySelector('.auth-tab.is-active').dataset.mode;
  const phone = $('phone').value.trim();
  const password = $('password').value;
  const nickname = $('nickname').value.trim();
  hide($('authError'));

  if (!/^1\d{10}$/.test(phone)) { showAuthError('请输入 11 位手机号'); return; }
  if (password.length < 6) { showAuthError('密码至少 6 位'); return; }
  if (mode === 'register' && !nickname) { showAuthError('请填写昵称'); return; }

  setAuthBusy(true);
  try {
    if (mode === 'register') await doRegister(phone, password, nickname);
    await doLogin(phone, password);
    enterApp();
  } catch (err) {
    showAuthError(err.message || '操作失败，请重试');
  } finally {
    setAuthBusy(false);
  }
}

async function validateSession() {
  const tok = localStorage.getItem(LS_TOKEN);
  const uid = localStorage.getItem(LS_USER);
  if (!tok || !uid || tok.length < 10) return false;
  try {
    const res = await fetch(API_BASE + '/api/v1/user/me', {
      headers: { Authorization: 'Bearer ' + tok },
    });
    if (!res.ok) return false;
    state.token = tok;
    state.userId = uid;
    return true;
  } catch (err) {
    return true; // 网络错误：保留登录状态，先进入界面
  }
}

/* ---------- 对话（SSE 流式） ---------- */
function handleSSEEvent(evt, full) {
  if (evt.type === 'progress') {
    const names = {
      input_check: '检查输入', mem_load: '加载记忆', intent_node: '识别意图',
      entities: '提取实体', knowledge: '检索知识', plan: '制定计划',
    };
    return { kind: 'progress', label: names[evt.node] || evt.node };
  }
  if (evt.type === 'intent') {
    updateDiagnostics({ intent: evt.intent, intent_analysis: evt.intent_analysis, target_agent: evt.target_agent });
    return { kind: 'intent', intent: evt.intent, confidence: (evt.intent_analysis || {}).confidence };
  }
  if (evt.type === 'chunk') { full.content += evt.content; return { kind: 'render' }; }
  if (evt.type === 'content') { full.content = evt.content; return { kind: 'render' }; }
  if (evt.type === 'done') {
    if (evt.session_id) state.sessionId = evt.session_id;
    updateDiagnostics({ conversation_turns: evt.conversation_turns, needs_confirmation: evt.needs_confirmation });
    refreshSessionDisplay();
    return { kind: 'done' };
  }
  if (evt.type === 'error') { full.content += evt.content; return { kind: 'render' }; }
  return { kind: 'none' };
}

async function sendMessage() {
  if (state.busy) return;
  const input = $('userInput');
  const text = input.value.trim();
  if (!text) return;

  addMessage('user', text);
  input.value = '';
  autogrow();

  const assistantEl = addMessage('assistant', '');
  assistantEl.classList.add('is-thinking');
  state.busy = true;
  setBusy(true);

  const full = { content: '' };
  const paint = (withCursor) => {
    assistantEl.classList.remove('is-thinking');
    assistantEl._bubble.innerHTML =
      renderMarkdown(full.content) + (withCursor ? '<span class="streaming-cursor"></span>' : '');
    scrollToBottom();
  };

  try {
    const res = await api('/api/v1/chat/completion', {
      method: 'POST',
      body: JSON.stringify({
        user_input: text,
        session_id: state.sessionId || '',
        stream: true,
      }),
    });

    if (!res.ok) {
      if (res.status === 401) { leaveApp(); showAuthError('登录已过期，请重新登录'); return; }
      throw new Error('请求失败 (' + res.status + ')');
    }

    const ctype = res.headers.get('content-type') || '';
    if (ctype.includes('text/event-stream') || res.body) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;
          let evt;
          try { evt = JSON.parse(jsonStr); } catch (err) { continue; }

          const r = handleSSEEvent(evt, full);
          if (r.kind === 'progress') {
            assistantEl._bubble.innerHTML =
              '<span style="color:var(--c-muted)">正在' + escapeHtml(r.label) + '…</span><span class="streaming-cursor"></span>';
            scrollToBottom();
          } else if (r.kind === 'intent') {
            setIntentChip(assistantEl, r.intent, r.confidence);
            paint(true);
          } else if (r.kind === 'render') {
            paint(true);
          } else if (r.kind === 'done') {
            paint(false);
          }
        }
      }
      if (full.content) paint(false);
    } else {
      const body = await res.json();
      const data = (body && body.data) || {};
      if (data.session_id) state.sessionId = data.session_id;
      full.content = data.assistant_output || '暂无回复';
      updateDiagnostics(data);
      refreshSessionDisplay();
      paint(false);
    }
  } catch (err) {
    console.error('发送失败:', err);
    assistantEl.classList.remove('is-thinking');
    assistantEl._bubble.textContent = '抱歉，系统暂时无法响应，请稍后再试。';
  } finally {
    state.busy = false;
    setBusy(false);
  }
}

/* ---------- 侧边栏（移动端） ---------- */
function openSidebar() {
  $('sidebar').classList.add('is-open');
  show($('sideOverlay'));
  $('menuBtn').setAttribute('aria-expanded', 'true');
}
function closeSidebar() {
  $('sidebar').classList.remove('is-open');
  hide($('sideOverlay'));
  $('menuBtn').setAttribute('aria-expanded', 'false');
}

/* ---------- 初始化 ---------- */
function setupAuthTabs() {
  document.querySelectorAll('.auth-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.auth-tab').forEach((t) => {
        t.classList.toggle('is-active', t === tab);
        t.setAttribute('aria-selected', String(t === tab));
      });
      const isRegister = tab.dataset.mode === 'register';
      $('nicknameField').hidden = !isRegister;
      $('authSubmit').textContent = isRegister ? '注 册' : '登 录';
      hide($('authError'));
    });
  });
}

function setupEvents() {
  $('authForm').addEventListener('submit', handleAuthSubmit);
  $('logoutBtn').addEventListener('click', leaveApp);
  $('sendBtn').addEventListener('click', sendMessage);
  $('newChatBtn').addEventListener('click', startNewChat);
  $('menuBtn').addEventListener('click', openSidebar);
  $('sideOverlay').addEventListener('click', closeSidebar);

  const ta = $('userInput');
  ta.addEventListener('input', autogrow);
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  $('suggestions').addEventListener('click', (e) => {
    const card = e.target.closest('.cap-card');
    if (!card) return;
    $('userInput').value = card.dataset.example;
    autogrow();
    $('userInput').focus();
  });
}

async function init() {
  setupAuthTabs();
  setupEvents();
  const ok = await validateSession();
  if (ok) enterApp();
}

document.addEventListener('DOMContentLoaded', init);
