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
  //: 待发送的图片附件 `{file, dataUrl, mime, name, size, previewUrl}`，或 null。
  //: 选图只挂在这里，**不发任何请求**；点发送时才随消息一起上去（见 attachLabImage）。
  //: 之所以存 dataUrl 而不是只存 File：发送要立刻用，而 FileReader 是异步的 ——
  //: 留到点击那一刻再读，用户会看到一个"点了没反应"的间隙。
  attachment: null,
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

/* ---------- 图片尺寸硬约束（行内 style + !important） ----------
 *
 * 为什么不能只靠 app.css 里的规则：样式表可能因缓存命中旧副本、或被第三方（浏览器扩展）
 * 注入的规则盖掉而失效 —— 这类"我们的规则没生效"从代码里完全看不出来，只在用户浏览器里发生。
 *
 * 行内 style + `!important` 是页面这一侧能拿到的最强声明：按 CSS 优先级，style 属性上的
 * !important 高于**任何**作者样式表的 !important，与选择器特异性无关。两道防线互补 ——
 * app.css 兜"JS 没跑到"，这里兜"CSS 没生效"。
 *
 * 约束用 `max-*` + `object-fit: contain`，不写死 width/height：缩略图永远装进框里、不裁切，
 * 横图/竖图/超长化验单都只是等比变小。 */
const IMG_FIT_MSG = {
  width: 'auto', height: 'auto',
  'max-width': 'min(240px, 100%)', 'max-height': '180px',
  'object-fit': 'contain',
};
const IMG_FIT_THUMB = {
  width: '52px', height: '40px',
  'max-width': '52px', 'max-height': '40px',
  'object-fit': 'cover',
};

/* ⚠️ 键名必须是 **kebab-case**（`max-width`），不能写 camelCase（`maxWidth`）：
   `CSSStyleDeclaration.setProperty` 不做驼峰转换，名字不认识就**静默忽略**（不抛异常、
   不返回 false），守卫"看起来跑了"实际只挂上一半。改这里的键名后，量一次
   `el.getAttribute('style')` 逐条核对。 */
function lockImageSize(img, fit) {
  if (!img || !img.style) return;
  Object.keys(fit).forEach((k) => {
    try { img.style.setProperty(k, fit[k], 'important'); } catch (e) { /* 忽略 */ }
  });
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
    // 气泡里如实放出用户发的图。"我发了这张图"必须在界面上看得见 ——
    // 只显示一句话的话，用户没法确认自己到底发出去了哪一张。
    if (meta && meta.imageUrl) {
      const img = document.createElement('img');
      img.className = 'msg-img';
      lockImageSize(img, IMG_FIT_MSG);
      img.src = meta.imageUrl;
      img.alt = '上传的图片';
      bubble.appendChild(img);
    }
    // 文本可能为空（只发图不打字）：此时不塞一个空文本节点，
    // 否则气泡里会多出一行空白把图片顶下去。
    if (content) {
      const textEl = document.createElement('div');
      textEl.className = 'msg-text';
      textEl.textContent = content;
      bubble.appendChild(textEl);
    }
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

/* ---------- 缓存命中统计（来自 done 事件） ---------- */
function updateCacheStats(cache) {
  if (!cache || cache.calls == null) return;
  const rate = cache.hit_rate != null ? Math.round(cache.hit_rate * 100) + '%' : '—';
  $('diagCacheRate').textContent = rate;
  $('diagCacheDetail').textContent =
    '命中 ' + (cache.cached_tokens ?? 0) + ' / 输入 ' + (cache.prompt_tokens ?? 0) +
    ' token · ' + cache.calls + ' 次调用' +
    (cache.hit_rate > 0.5 ? '' : '（缓存预热中，随对话推进上升）');
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
  $('attachBtn').disabled = b;
  document.querySelectorAll('.cap-card').forEach((c) => { c.disabled = b; });
  // 只动当前有效的那张选择卡：过期卡（is-stale）必须保持禁用，否则请求结束后被
  // 这里一并解锁，用户点了就会把旧选项答到新的待确认记录上。
  document.querySelectorAll('.reply-chip:not(.is-stale)').forEach((c) => { c.disabled = b; });
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
  clearAttachStrip();
  show($('emptyState'));
  $('diagTurns').textContent = '0';
  $('diagIntent').textContent = '—';
  $('diagTarget').textContent = '—';
  $('diagConfidence').textContent = '—';
  $('diagCacheRate').textContent = '—';
  $('diagCacheDetail').textContent = '—';
  $('diagReason').textContent = '—';
  refreshSessionDisplay();
  hide($('appView'));
  show($('authView'));
}

function startNewChat() {
  state.sessionId = null;
  state.turns = 0;
  $('messageList').innerHTML = '';
  clearAttachStrip();
  show($('emptyState'));
  $('diagTurns').textContent = '0';
  $('diagIntent').textContent = '—';
  $('diagTarget').textContent = '—';
  $('diagConfidence').textContent = '—';
  $('diagCacheRate').textContent = '—';
  $('diagCacheDetail').textContent = '—';
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
    if (!res.ok) return false; // token 无效/过期 → 回登录页
    state.token = tok;
    state.userId = uid;
    return true;
  } catch (err) {
    // 网络错误（后端不可达）时不再盲目自动登录：只有验证通过才进入应用，
    // 否则会用过期 token 进入聊天页再撞 401，造成"自动登录后又报错"的困惑。
    return false;
  }
}

/* ---------- 对话（SSE 流式） ---------- */
function handleSSEEvent(evt, full) {
  if (evt.type === 'progress') {
    // 节点名 → 中文标签。后端 progress 事件直接来自图节点名（Step 3 起由
    // graph.astream(stream_mode="updates") 自动给出），新增节点务必在此登记，
    // 否则会直接显示英文标识符。
    const names = {
      input_check: '检查输入', mem_load: '加载记忆', intent_node: '识别意图',
      entities: '提取实体', knowledge: '检索知识', plan: '制定计划',
      execute: '执行计划', reconcile: '汇总结果', llm: '生成回答',
      fact_check: '核对事实', out: '输出检查', commit: '提交结果',
      mem: '更新记忆', err: '处理异常',
      // 不是图节点：由 chat_router 在识别图片时直接推（图还没建好，界面不能干等）
      lab_vision: '识别化验单',
    };
    return { kind: 'progress', label: names[evt.node] || evt.node };
  }
  if (evt.type === 'intent') {
    updateDiagnostics({ intent: evt.intent, intent_analysis: evt.intent_analysis, target_agent: evt.target_agent });
    return { kind: 'intent', intent: evt.intent, confidence: (evt.intent_analysis || {}).confidence };
  }
  if (evt.type === 'chunk') { full.content += evt.content; return { kind: 'render' }; }
  if (evt.type === 'content') { full.content = evt.content; return { kind: 'render' }; }
  if (evt.type === 'options') {
    // 写操作二次确认的选择卡。按钮要挂在**这一条**助手消息下面，而事件在文本吐完
    // 之前就到了，所以先存进 full，等本轮流结束再渲染（见 flushQuickReplies）。
    full.options = evt.options || [];
    return { kind: 'none' };
  }
  if (evt.type === 'done') {
    if (evt.session_id) state.sessionId = evt.session_id;
    updateDiagnostics({ conversation_turns: evt.conversation_turns, needs_confirmation: evt.needs_confirmation });
    if (evt.cache) updateCacheStats(evt.cache);
    refreshSessionDisplay();
    return { kind: 'done' };
  }
  if (evt.type === 'error') { full.content += evt.content; return { kind: 'render' }; }
  return { kind: 'none' };
}

function sendMessage() {
  if (state.busy) return;
  const input = $('userInput');
  const text = input.value.trim();
  // 只发一张图、一个字都不打也是合法的一轮：判断条件是"有文本**或**有附件"。
  if (!text && !state.attachment) return;
  input.value = '';
  autogrow();
  const att = state.attachment;
  // 把附件从 state 摘走再交给 sendUserText：预览 URL 的所有权随之转移给那条消息气泡，
  // 免得 sendUserText 里 clearAttachStrip() 把气泡正要用的图撤销掉。
  state.attachment = null;
  return sendUserText(text, att);
}

/* 发送一段文本（可带一张待发附件）。从输入框发送（sendMessage）和点确认卡按钮
 * （quickReply）共用这一条路径 —— 选择卡的按钮本质上就是"替用户打了一句话"，
 * 走同一入口才不会有第二套行为。quickReply 不传附件：那一下是在回答确认卡，
 * 把图片捎进去只会让后端把它当成同一轮的问题。
 */
async function sendUserText(text, attachment) {
  if (state.busy) return;
  if (!text && !attachment) return;

  addMessage('user', text, attachment ? { imageUrl: attachment.previewUrl } : undefined);
  if (attachment) {
    // 图片已经进了气泡，预览条不再需要（URL 留着给气泡用，见 clearAttachStrip 的说明）
    clearAttachStrip();
  }

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
    const payload = {
      user_input: text,
      session_id: state.sessionId || '',
      stream: true,
    };
    if (attachment) {
      // 图片与文本一起进对话，由**后端**识别并结合文本作答。
      // 服务端会先推一条 progress(lab_vision)，前端显示"正在识别化验单"。
      payload.image_base64 = attachment.dataUrl;
      payload.image_mime = attachment.mime;
    }
    const res = await api('/api/v1/chat/completion', {
      method: 'POST',
      body: JSON.stringify(payload),
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
      flushQuickReplies(assistantEl, full);
    } else {
      const body = await res.json();
      const data = (body && body.data) || {};
      if (data.session_id) state.sessionId = data.session_id;
      full.content = data.assistant_output || '暂无回复';
      full.options = data.options || [];
      updateDiagnostics(data);
      if (data.cache_stats) updateCacheStats(data.cache_stats);
      refreshSessionDisplay();
      paint(false);
      flushQuickReplies(assistantEl, full);
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

/* ---------- 写操作二次确认的选项卡 ----------
 * 点击后按普通消息再发一轮，发的是 **label 而不是 id**：后端
 * drug_write_confirmation.resolve_answer 有 label 精确匹配分支，而且 label 本身
 * 就是合法药名/动作词（"布洛芬" / "不记录"），即便那条分支漏了也能靠自由文本兜底。
 *
 * 同一时刻只允许最后一张卡可点：pending_confirmation 在会话里是**单槽**的，新一轮
 * 确认会覆盖上一张卡。若旧卡按钮仍可点，用户点它等于把一个过期选项答到当前待确认的
 * 记录上——正是这个功能要防的事，所以旧卡一律置灰（is-stale）。
 */
function flushQuickReplies(assistantEl, full) {
  const options = full.options || [];
  if (!options.length) return;

  document.querySelectorAll('.reply-chip:not(.is-stale)').forEach((btn) => {
    btn.classList.add('is-stale');
    btn.disabled = true;
    btn.title = '这张确认卡已过期，请使用最新的那张';
  });

  const body = assistantEl.querySelector('.msg-body');
  if (!body) return;
  const row = document.createElement('div');
  row.className = 'reply-row';
  options.forEach((opt) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'reply-chip';
    btn.textContent = opt.label;
    btn.dataset.reply = opt.label;
    row.appendChild(btn);
  });
  body.appendChild(row);
  scrollToBottom();
}

/* ---------- 化验单图片识别（图 → 可编辑文本，不产出结论） ----------
 * 设计约束：识别结果**只回填到输入框**，绝不自动发送。
 * 用户先看到逐条结果、改了再发，走的还是和手工输入完全相同的那条链路 ——
 * 所以"图路"和"文路"的判定口径不可能分叉。
 * 后端在识别阶段就挡掉了单位冲突 / 比较符值（见 lab_report_vision.py），
 * 这里只负责展示与被挡住的原因，不重复判断。
 */
const LAB_IMAGE_MAX_MB = 5; // 仅前置提示；真正口径在后端 settings.LAB_IMAGE_MAX_MB
// 与后端 `_ALLOWED_FORMATS` 对齐。⚠️ 不含 HEIC/HEIF —— 后端无 pillow-heif，确实读不了；
// iPhone 默认拍照就是 HEIC，所以宁可在这里就明确报错，也不要传上去再失败。
const LAB_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/avif'];

function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result || ''));
    fr.onerror = () => reject(new Error('读取文件失败'));
    fr.readAsDataURL(file);
  });
}

function setAttachStatus(text, kind) {
  const strip = $('attachStrip');
  strip.classList.remove('is-ok', 'is-error');
  if (kind === 'ok') strip.classList.add('is-ok');
  if (kind === 'error') strip.classList.add('is-error');
  $('attachStatus').textContent = text;
}

function showAttachStrip(att) {
  const thumb = $('attachThumb');
  // 预览条缩略图同样写行内约束：它比气泡里的图更早出现（选完文件就渲染），
  // 一旦 app.css 没生效，用户第一眼看到的就是"图片撑满页面"。
  lockImageSize(thumb, IMG_FIT_THUMB);
  thumb.src = att.previewUrl || '';
  $('attachName').textContent =
    att.name + ' · ' + Math.max(1, Math.round(att.size / 1024)) + ' KB';
  show($('attachStrip'));
}

function clearAttachStrip() {
  if (state.attachment) {
    // 预览 URL 在这里释放。⚠️ 已经发出去的那条消息气泡里用的是**同一个** URL
    // （见 sendUserText 的转移说明），所以发送时会把它从 state.attachment 摘走，
    // 不会走到这里被撤销。
    URL.revokeObjectURL(state.attachment.previewUrl);
    state.attachment = null;
  }
  $('attachThumb').removeAttribute('src');
  hide($('attachStrip'));
  setAttachStatus('—');
  $('labImageInput').value = '';
}

/* 选图 = 挂成**本地待发送附件**，不联网。
 *
 * 为什么不在选图时就把请求发出去（改造前的做法）：
 *   1) 识别结果曾被直接写进输入框，等于**把系统的解析产物冒充成用户打的字** ——
 *      输入框、以及随后的对话记录里都会记成是他说的，对话记录不再可信；
 *   2) 图片和用户的问题被拆成两轮，用户想问"这张单子严重吗"只能先发一次识别文本；
 *   3) 只是选错了文件或想先看一眼，付费的视觉识别已经跑完了。
 * 现在：选图 → 挂在这里 → 想打字就打字 → 点发送，图片与文本一起进对话，
 * 由后端识别并结合文本作答（`/api/v1/chat/completion` 的 image_base64 / image_mime）。
 * 单独的 `/api/v1/lab/image-extract` 接口仍在（它被聊天路径复用），只是前端不再调它。
 */
async function attachLabImage(file) {
  if (!file || state.busy) return;

  if (!LAB_IMAGE_TYPES.includes(file.type)) {
    setAttachStatus(
      '不支持的图片格式（' + (file.type || '未知') + '），请使用 JPG / PNG / WebP / AVIF', 'error');
    return;
  }
  if (file.size > LAB_IMAGE_MAX_MB * 1024 * 1024) {
    setAttachStatus('图片超过 ' + LAB_IMAGE_MAX_MB + 'MB，请压缩后再试', 'error');
    return;
  }

  if (state.attachment) URL.revokeObjectURL(state.attachment.previewUrl);
  state.attachment = { file: file, mime: file.type, name: file.name, size: file.size, previewUrl: null };

  try {
    // base64 直接经 JSON 发送：与本项目其余接口同一套约定，
    // 也省掉 multipart（后端未依赖 python-multipart）。格式真伪由后端解码判定。
    state.attachment.dataUrl = await readAsDataURL(file);
  } catch (err) {
    console.error('读取图片失败:', err);
    state.attachment = null;
    clearAttachStrip();
    setAttachStatus('读取图片失败，请重试', 'error');
    return;
  }

  state.attachment.previewUrl = URL.createObjectURL(file);
  showAttachStrip(state.attachment);
  // 打不打字都可以：只发图也能发（sendMessage 的判断是"有文本或有附件"）
  setAttachStatus('将随消息一起发送，也可以再说一句话', 'ok');
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

  $('attachBtn').addEventListener('click', () => { if (!state.busy) $('labImageInput').click(); });
  $('attachRemove').addEventListener('click', clearAttachStrip);
  $('labImageInput').addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = ''; // 清空以便连续选同一张图仍能触发 change
    if (f) attachLabImage(f);
  });

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

  // 确认卡按钮用事件委托：卡片是流式渲染中动态插进消息列表的，逐个绑定会漏。
  $('messageList').addEventListener('click', (e) => {
    const btn = e.target.closest('.reply-chip');
    if (!btn || btn.disabled) return;
    const reply = btn.dataset.reply;
    if (reply) sendUserText(reply);
  });
}

async function init() {
  setupAuthTabs();
  setupEvents();
  const ok = await validateSession();
  if (ok) enterApp();
}

document.addEventListener('DOMContentLoaded', init);
