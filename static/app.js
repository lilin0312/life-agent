/**
 * 生活管家 AI-Agent 前端应用
 * 功能：聊天、语音、会话管理、文件上传、记忆管理、图片生成、面板工具
 */
class LifeAgent {
    constructor() {
        this.sessionId = null;
        this.isLoading = false;
        this.apiBase = '/api';
        this.pendingImage = null;
        this.isRecording = false;
        this.recognition = null;
        this.abortController = null;  // 用于取消正在进行的请求

        // ---- 通话相关属性 ----
        this.callActive = false;
        this.callStartTime = null;
        this.callTimerInterval = null;
        this.isCallRecording = false;
        this.isAiSpeaking = false;
        this.isMuted = false;
        this.callRecognizedText = '';
        this.currentUtterance = null; // SpeechSynthesisUtterance
        this.receivedAudioTotal = 0;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadSessions();
        this.autoResizeInput();
        this.initVoice();
    }

    bindElements() {
        this.messagesEl = document.getElementById('messages');
        this.inputEl = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.sessionListEl = document.getElementById('sessionList');
        this.userIdEl = document.getElementById('userId');
        this.fileInput = document.getElementById('fileInput');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreviewBar = document.getElementById('imagePreviewBar');
        this.imagePreviewThumb = document.getElementById('imagePreviewThumb');
        this.imagePreviewName = document.getElementById('imagePreviewName');
        this.voiceBtn = document.getElementById('voiceBtn');
    }

    bindEvents() {
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
    }

    get userId() {
        return this.userIdEl.value.trim() || 'user_001';
    }

    // ==================== 消息发送 ====================

    async sendMessage(text) {
        const message = text || this.inputEl.value.trim();
        if (!message || this.isLoading) return;

        this.isLoading = true;
        this.inputEl.value = '';
        this.inputEl.style.height = 'auto';
        this.showCancelButton(true);

        this.clearWelcome();
        this.appendMessage('user', message);

        if (this.pendingImage) {
            const lastMsg = this.messagesEl.querySelector('.message.user:last-child .bubble');
            if (lastMsg) {
                const img = document.createElement('img');
                img.src = this.pendingImage;
                img.style.cssText = 'max-width:200px;max-height:150px;border-radius:8px;margin-top:6px;display:block;';
                lastMsg.appendChild(img);
            }
        }

        const loadingId = this.showLoading();
        this.abortController = new AbortController();

        try {
            const startTime = Date.now();
            const body = {
                user_id: this.userId,
                message: message,
                session_id: this.sessionId,
            };
            if (this.pendingImage) {
                body.image_base64 = this.pendingImage;
            }
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: this.abortController.signal,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.session_id) {
                this.sessionId = data.session_id;
                this.loadSessions();
            }

            this.removeLoading(loadingId);

            if (data.need_confirm && data.pending_id) {
                this.showConfirmDialog(data.pending_id, data.confirm_preview || data.content);
            }

            this.appendMessage('ai', data.content, elapsed, data.tool_used);

        } catch (error) {
            this.removeLoading(loadingId);
            if (error.name === 'AbortError') {
                this.appendMessage('system', '已取消发送');
            } else {
                this.appendMessage('ai', `⚠️ 请求失败: ${error.message}\n请检查网络连接或稍后重试。`);
            }
        } finally {
            this.isLoading = false;
            this.abortController = null;
            this.showCancelButton(false);
            this.clearImage();
            this.inputEl.focus();
        }
    }

    // ==================== 消息渲染 ====================

    clearWelcome() {
        const welcome = this.messagesEl.querySelector('.welcome-message');
        if (welcome) welcome.remove();
    }

    appendMessage(role, content, elapsed, toolUsed) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        if (role === 'system') {
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = content;
            div.appendChild(bubble);
            this.messagesEl.appendChild(div);
            this.scrollToBottom();
            return;
        }

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = role === 'user' ? '👤' : '🏠';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        let html = this.renderMarkdown(content);

        if (toolUsed) {
            const toolNames = {
                calculator: '🔢 计算器',
                get_current_time: '🕐 时间查询',
                save_memory: '🧠 记忆保存',
                search_memory: '🧠 记忆搜索',
                search_documents: '📄 文档检索',
                generate_image: '🎨 图片生成',
                get_weather: '🌤️ 天气查询',
                translate: '🌐 翻译',
                web_search: '🔍 网页搜索',
                analyze_image: '🖼️ 图片分析',
            };
            const toolList = toolUsed.split(', ').map(t => toolNames[t] || t).join(', ');
            html = `<div style="font-size:11px;color:var(--text-light);margin-bottom:6px;">使用了 ${toolList}</div>` + html;
        }

        if (elapsed) {
            html += `<div style="font-size:11px;color:var(--text-light);margin-top:6px;text-align:right;">${elapsed}s</div>`;
        }

        bubble.innerHTML = html;
        div.appendChild(avatar);
        div.appendChild(bubble);
        this.messagesEl.appendChild(div);
        this.scrollToBottom();
    }

    showLoading() {
        const id = 'loading-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message ai';
        div.id = id;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = '🏠';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;

        div.appendChild(avatar);
        div.appendChild(bubble);
        this.messagesEl.appendChild(div);
        this.scrollToBottom();
        return id;
    }

    cancelRequest() {
        if (this.abortController) {
            this.abortController.abort();
        }
    }

    showCancelButton(show) {
        if (show) {
            this.sendBtn.disabled = true;
            this.sendBtn.textContent = '⏹ 停止';
            this.sendBtn.disabled = false;
            this.sendBtn.onclick = () => this.cancelRequest();
        } else {
            this.sendBtn.textContent = '➤';
            this.sendBtn.onclick = () => this.sendMessage();
        }
    }

    removeLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    renderMarkdown(text) {
        if (!text) return '';

        // 检测聊天历史中的图片标记 [IMG]path[/IMG]
        let imageHtml = '';
        text = text.replace(/\[IMG\](.*?)\[\/IMG\]/g, (match, path) => {
            imageHtml += `<img src="${path}" style="max-width:300px;max-height:300px;border-radius:8px;margin:4px 0;display:block;" loading="lazy" />`;
            return '';  // 移除标记，不显示在文字中
        });

        let html = imageHtml + text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // 图片链接 ![alt](url)
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;border-radius:8px;margin:8px 0;" />');
        // 普通链接 [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:var(--primary);">$1</a>');
        // **bold**
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // *italic*
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        // `code`
        html = html.replace(/`(.+?)`/g, '<code style="background:#e2e8f0;padding:2px 6px;border-radius:3px;font-size:13px;">$1</code>');
        // 列表项
        html = html.replace(/^(\d+)\.\s/gm, '<br>$1. ');
        html = html.replace(/^[-*]\s/gm, '<br>• ');

        return html;
    }

    scrollToBottom() {
        requestAnimationFrame(() => {
            this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        });
    }

    // ==================== 会话管理 ====================

    async loadSessions() {
        try {
            const resp = await fetch(`${this.apiBase}/sessions/${this.userId}`);
            const data = await resp.json();
            if (data.success && data.sessions.length) {
                this.renderSessions(data.sessions);
            }
        } catch (e) { /* 静默 */ }
    }

    renderSessions(sessions) {
        this.sessionListEl.innerHTML = '';
        sessions.forEach(s => {
            const div = document.createElement('div');
            div.className = 'session-item' + (s.session_id === this.sessionId ? ' active' : '');
            const date = new Date(s.updated_at).toLocaleString('zh-CN', {
                month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            const label = document.createElement('span');
            label.textContent = `💬 ${date}`;
            label.style.flex = '1';
            label.onclick = () => this.switchSession(s.session_id);

            const delBtn = document.createElement('button');
            delBtn.className = 'session-delete';
            delBtn.textContent = '✕';
            delBtn.title = '删除会话';
            delBtn.onclick = async (e) => {
                e.stopPropagation();
                if (confirm('确定删除这个会话？')) {
                    await fetch(`${this.apiBase}/admin/session/${s.session_id}`, { method: 'DELETE' });
                    if (this.sessionId === s.session_id) {
                        this.newChat();
                    }
                    this.loadSessions();
                }
            };

            div.appendChild(label);
            div.appendChild(delBtn);
            this.sessionListEl.appendChild(div);
        });
    }

    async switchSession(sessionId) {
        this.sessionId = sessionId;
        try {
            const resp = await fetch(`${this.apiBase}/history/${sessionId}`);
            const data = await resp.json();
            if (data.success) {
                this.messagesEl.innerHTML = '';
                data.history.forEach(msg => {
                    this.appendMessage(msg.role, msg.content);
                });
            }
        } catch (e) {
            console.error('加载历史失败:', e);
        }
        this.loadSessions();
    }

    newChat() {
        this.sessionId = null;
        this.messagesEl.innerHTML = '';
        this.inputEl.focus();
        this.loadSessions();
    }

    // ==================== 文件上传 ====================

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.showToast(`正在上传 ${file.name}...`, 'warning');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', this.userId);

        try {
            const resp = await fetch(`${this.apiBase}/upload`, {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();
            if (resp.ok && data.success) {
                this.showToast(`✅ ${data.message}`, 'success');
            } else {
                this.showToast(`❌ ${data.detail || data.message || '上传失败'}`, 'error');
            }
        } catch (e) {
            this.showToast(`❌ 上传出错: ${e.message}`, 'error');
        }
        event.target.value = '';
    }

    // ==================== 图片上传 ====================

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.pendingImage = e.target.result;
                this.imagePreviewThumb.src = this.pendingImage;
                this.imagePreviewName.textContent = file.name;
                this.imagePreviewBar.style.display = 'flex';
                this.showToast(`🖼️ 已添加图片: ${file.name}，输入问题后发送`, 'success');
            };
            reader.readAsDataURL(file);
        } else {
            const reader = new FileReader();
            reader.onload = (e) => {
                const text = e.target.result.substring(0, 3000);
                this.inputEl.value = `请分析以下文件内容（${file.name}）：\n\`\`\`\n${text}\n\`\`\``;
                this.inputEl.style.height = 'auto';
                this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + 'px';
                this.showToast(`📄 已读取文件: ${file.name}`, 'success');
            };
            reader.readAsText(file);
        }
        event.target.value = '';
    }

    clearImage() {
        this.pendingImage = null;
        this.imagePreviewBar.style.display = 'none';
    }

    // ==================== 语音识别 ====================

    initVoice() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            this.voiceBtn.title = '浏览器不支持语音识别（请用Chrome）';
            this.voiceBtn.style.opacity = '0.4';
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.lang = 'zh-CN';
        this.recognition.continuous = false;
        this.recognition.interimResults = true;

        this.recognition.onresult = (e) => {
            let text = '';
            for (let i = e.resultIndex; i < e.results.length; i++) {
                text += e.results[i][0].transcript;
            }
            this.inputEl.value = text;
            this.inputEl.style.height = 'auto';
            this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + 'px';
        };

        this.recognition.onend = () => {
            this.isRecording = false;
            this.voiceBtn.classList.remove('recording');
            this.voiceBtn.textContent = '🎤';
        };

        this.recognition.onerror = (e) => {
            this.isRecording = false;
            this.voiceBtn.classList.remove('recording');
            this.voiceBtn.textContent = '🎤';
            if (e.error !== 'no-speech') {
                this.showToast(`语音识别出错: ${e.error}`, 'error');
            }
        };
    }

    toggleVoice() {
        if (!this.recognition) {
            this.showToast('浏览器不支持语音识别，请使用 Chrome', 'error');
            return;
        }

        if (this.isRecording) {
            this.recognition.stop();
            this.isRecording = false;
            this.voiceBtn.classList.remove('recording');
            this.voiceBtn.textContent = '🎤';
        } else {
            this.recognition.start();
            this.isRecording = true;
            this.voiceBtn.classList.add('recording');
            this.voiceBtn.textContent = '⏺';
            this.showToast('🎙️ 正在录音...请说话', 'success');
        }
    }

    // ==================== 面板管理 ====================

    async showPanel(panelId) {
        // 关闭其他面板
        document.querySelectorAll('.side-panel.open').forEach(p => {
            if (p.id !== panelId) p.classList.remove('open');
        });

        const panel = document.getElementById(panelId);
        panel.classList.toggle('open');

        if (panel.classList.contains('open')) {
            await this.loadPanelData(panelId);
        }
    }

    closePanel(panelId) {
        document.getElementById(panelId).classList.remove('open');
    }

    async loadPanelData(panelId) {
        const bodyEl = document.getElementById(panelId + 'Body');
        if (!bodyEl) return;

        switch (panelId) {
            case 'memoryPanel':
                await this.loadMemoryPanel(bodyEl);
                break;
            case 'vectorPanel':
                await this.loadVectorPanel(bodyEl);
                break;
            case 'dbPanel':
                await this.loadDbPanel(bodyEl);
                break;
        }
    }

    async loadMemoryPanel(el) {
        try {
            const resp = await fetch(`${this.apiBase}/admin/all-memories/${this.userId}`);
            const data = await resp.json();

            if (!data.success || !data.memories.length) {
                el.innerHTML = '<div class="panel-empty"><div class="empty-icon">🧠</div>暂无记忆<br><small>在对话中告诉AI你的信息即可自动保存</small></div>';
                return;
            }

            el.innerHTML = data.memories.map(m => `
                <div class="panel-item">
                    <div class="panel-item-header">
                        <span class="panel-item-key">🔑 ${m.mem_key}</span>
                        <span class="panel-item-time">${new Date(m.updated_at).toLocaleString('zh-CN')}</span>
                    </div>
                    <div class="panel-item-content">${m.content}</div>
                    <div class="panel-item-actions">
                        <button class="btn-panel-sm danger" onclick="app.deleteMemory('${m.mem_key}')">删除</button>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            el.innerHTML = `<div class="panel-empty">加载失败: ${e.message}</div>`;
        }
    }

    async deleteMemory(key) {
        if (!confirm(`确定删除记忆「${key}」？`)) return;
        await fetch(`${this.apiBase}/memory/${this.userId}/${key}`, { method: 'DELETE' });
        this.showToast(`已删除: ${key}`, 'success');
        this.loadPanelData('memoryPanel');
    }

    async loadVectorPanel(el) {
        try {
            const resp = await fetch(`${this.apiBase}/admin/vectordb`);
            const data = await resp.json();
            const info = data.info;

            let html = `<div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-num">${info.ready ? '✅' : '❌'}</div>
                    <div class="stat-label">服务状态</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">${info.files.length}</div>
                    <div class="stat-label">上传文件数</div>
                </div>
            </div>`;

            if (info.vectordb_size) {
                const sizeMB = (info.vectordb_size / 1024 / 1024).toFixed(2);
                html += `<div style="font-size:13px;color:var(--text-light);margin-bottom:12px;">向量库大小: ${sizeMB} MB</div>`;
            }

            if (info.files.length) {
                html += '<div style="font-size:12px;font-weight:600;margin-bottom:8px;">📁 已上传文件</div>';
                info.files.forEach(f => {
                    const sizeStr = f.size > 1024 ? (f.size / 1024).toFixed(1) + 'KB' : f.size + 'B';
                    html += `<div class="file-item">
                        <span class="file-icon">📄</span>
                        <span class="file-name">${f.name}</span>
                        <span class="file-size">${sizeStr}</span>
                    </div>`;
                });
            } else {
                html += '<div class="panel-empty"><div class="empty-icon">📊</div>暂无文档<br><small>点击顶部「📎 文档入库」上传</small></div>';
            }

            el.innerHTML = html;
        } catch (e) {
            el.innerHTML = `<div class="panel-empty">加载失败: ${e.message}</div>`;
        }
    }

    async loadDbPanel(el) {
        try {
            const [statsResp, sessionsResp] = await Promise.all([
                fetch(`${this.apiBase}/admin/db-stats`),
                fetch(`${this.apiBase}/admin/all-sessions/${this.userId}`),
            ]);
            const stats = await statsResp.json();
            const sessions = await sessionsResp.json();

            let html = `<div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-num">${stats.stats.sessions || 0}</div>
                    <div class="stat-label">会话数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">${stats.stats.chat_history || 0}</div>
                    <div class="stat-label">消息总数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">${stats.stats.user_memory || 0}</div>
                    <div class="stat-label">记忆条数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">${stats.stats.pending_actions || 0}</div>
                    <div class="stat-label">待确认操作</div>
                </div>
            </div>`;

            if (sessions.success && sessions.sessions.length) {
                html += '<div style="font-size:12px;font-weight:600;margin-bottom:8px;">💬 会话列表</div>';
                sessions.sessions.forEach(s => {
                    const date = new Date(s.updated_at).toLocaleString('zh-CN');
                    html += `<div class="panel-item">
                        <div class="panel-item-header">
                            <span class="panel-item-key">${s.session_id.substring(0, 12)}...</span>
                            <span class="panel-item-time">${date}</span>
                        </div>
                        <div class="panel-item-content">${s.msg_count} 条消息</div>
                    </div>`;
                });
            }

            el.innerHTML = html;
        } catch (e) {
            el.innerHTML = `<div class="panel-empty">加载失败: ${e.message}</div>`;
        }
    }

    // ==================== 图片生成 ====================

    async generateImage() {
        const prompt = document.getElementById('imgPrompt').value.trim();
        const size = document.getElementById('imgSize').value;
        if (!prompt) {
            this.showToast('请输入图片描述', 'warning');
            return;
        }

        const resultEl = document.getElementById('imgGenResult');
        resultEl.innerHTML = '<div class="panel-loading">🎨 正在生成图片，请稍候...</div>';

        try {
            const resp = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: this.userId,
                    message: `请用 generate_image 工具帮我生成图片：${prompt}，尺寸 ${size}`,
                    session_id: this.sessionId,
                }),
            });
            const data = await resp.json();

            if (data.success) {
                // 从回复中提取图片URL
                const imgMatch = data.content.match(/!\[.*?\]\((https?:\/\/[^\s)]+)\)/);
                const urlMatch = data.content.match(/图片链接:\s*(https?:\/\/[^\s]+)/);

                if (imgMatch) {
                    resultEl.innerHTML = `<img src="${imgMatch[1]}" alt="${prompt}" />`;
                } else if (urlMatch) {
                    resultEl.innerHTML = `<img src="${urlMatch[1]}" alt="${prompt}" />`;
                } else {
                    resultEl.innerHTML = `<div class="panel-item-content">${data.content}</div>`;
                }

                if (data.session_id) {
                    this.sessionId = data.session_id;
                    this.loadSessions();
                }
            } else {
                resultEl.innerHTML = `<div class="panel-empty">生成失败: ${data.content}</div>`;
            }
        } catch (e) {
            resultEl.innerHTML = `<div class="panel-empty">请求失败: ${e.message}</div>`;
        }
    }

    // ==================== 工具方法 ====================

    showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        setTimeout(() => { toast.classList.remove('show'); }, 3000);
    }

    autoResizeInput() {
        this.inputEl.addEventListener('input', () => {
            this.inputEl.style.height = 'auto';
            this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + 'px';
        });
    }

    // ==================== 人机确认弹窗 ====================

    showConfirmDialog(pendingId, preview) {
        const overlay = document.createElement('div');
        overlay.className = 'confirm-overlay';
        overlay.id = 'confirm-' + pendingId;

        const box = document.createElement('div');
        box.className = 'confirm-box';
        box.innerHTML = `
            <div class="confirm-title">⚠️ 操作确认</div>
            <div class="confirm-body">${this.renderMarkdown(preview)}</div>
            <div class="confirm-actions">
                <button class="btn-confirm-reject" onclick="app.handleConfirm('${pendingId}', 'reject')">取消</button>
                <button class="btn-confirm-ok" onclick="app.handleConfirm('${pendingId}', 'confirm')">确认执行</button>
            </div>
        `;

        overlay.appendChild(box);
        document.body.appendChild(overlay);
    }

    async handleConfirm(pendingId, action) {
        const overlay = document.getElementById('confirm-' + pendingId);
        if (overlay) overlay.remove();

        if (action === 'reject') {
            this.appendMessage('system', '❌ 已取消操作');
            await fetch(`${this.apiBase}/confirm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pending_id: pendingId, user_id: this.userId, action: 'reject' }),
            });
            return;
        }

        this.appendMessage('system', '⏳ 正在执行...');
        try {
            const resp = await fetch(`${this.apiBase}/confirm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pending_id: pendingId, user_id: this.userId, action: 'confirm' }),
            });
            const data = await resp.json();
            if (data.success) {
                this.appendMessage('ai', '✅ ' + data.content);
            } else {
                this.appendMessage('ai', '⚠️ ' + data.content);
            }
        } catch (e) {
            this.appendMessage('ai', `❌ 执行失败: ${e.message}`);
        }
    }

    // ==================== 语音通话 ====================

    async toggleCall() {
        if (this.callActive) {
            this.hangUp();
        } else {
            await this.startCall();
        }
    }

    async startCall() {
        // 确保 SpeechRecognition 已初始化
        this._ensureRecognition();

        // 预加载 SpeechSynthesis 语音列表
        if (window.speechSynthesis) {
            window.speechSynthesis.getVoices();
            // Chrome 需要异步获取
            window.speechSynthesis.onvoiceschanged = () => {
                window.speechSynthesis.getVoices();
            };
        }

        // 显示通话界面
        document.getElementById('callOverlay').style.display = 'flex';
        document.getElementById('callBtn').classList.add('calling');
        document.getElementById('callBtn').textContent = '📞 通话中...';
        this.callActive = true;
        this.isMuted = false;
        this.callRecognizedText = '';

        this.updateCallStatus('listening');
        document.getElementById('callTranscript').innerHTML = '';
        document.getElementById('callTimer').textContent = '00:00';

        // 开始计时
        this.callStartTime = Date.now();
        this.updateCallTimer();
        this.callTimerInterval = setInterval(() => this.updateCallTimer(), 1000);

        this.showToast('📞 通话已就绪，按住中间按钮说话', 'success');
    }

    _ensureRecognition() {
        // 如果已有 recognition 实例且可用，直接返回
        if (this.recognition) return;

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.lang = 'zh-CN';
            this.recognition.continuous = false;
            this.recognition.interimResults = true;
        }
    }

    hangUp() {
        this.endCall();
    }

    endCall() {
        this.callActive = false;

        // 停止识别
        if (this.recognition) {
            try { this.recognition.abort(); } catch (e) { /* ignore */ }
        }
        this.isCallRecording = false;

        // 停止 TTS
        this.stopAiSpeaking();

        // 清除计时器
        if (this.callTimerInterval) {
            clearInterval(this.callTimerInterval);
            this.callTimerInterval = null;
        }

        // 隐藏通话界面
        document.getElementById('callOverlay').style.display = 'none';
        document.getElementById('callBtn').classList.remove('calling');
        document.getElementById('callBtn').textContent = '📞 打电话';
        document.getElementById('speakBtn').classList.remove('recording');

        this.loadSessions();
        // 恢复聊天语音识别回调
        this.initVoice();
        this.showToast('📞 通话已结束', 'success');
    }

    startSpeaking() {
        if (!this.callActive || this.isCallRecording) return;

        // 打断 AI 说话
        if (this.isAiSpeaking) {
            this.interruptAi();
        }

        // 使用浏览器原生 SpeechRecognition（已在 initVoice 中初始化）
        if (!this.recognition) {
            this.addTranscript('⚠️ 浏览器不支持语音识别，请用 Chrome', 'system');
            return;
        }

        try {
            this.isCallRecording = true;
            this.callRecognizedText = '';
            document.getElementById('speakBtn').classList.add('recording');
            document.getElementById('callWave').classList.add('active');
            this.updateCallStatus('listening');

            // 复用已有的 recognition 实例，但换用通话模式的回调
            this.recognition.continuous = false;
            this.recognition.interimResults = true;

            this.recognition.onresult = (e) => {
                let text = '';
                for (let i = e.resultIndex; i < e.results.length; i++) {
                    text += e.results[i][0].transcript;
                }
                this.callRecognizedText = text;
                // 实时显示识别中
                this.addTranscript(text, 'user-draft');
            };

            this.recognition.onend = () => {
                // 录音结束后自动处理
                if (this.isCallRecording) {
                    this._handleCallInput();
                }
            };

            this.recognition.onerror = (e) => {
                console.error('[Call] 识别错误:', e.error);
                if (e.error === 'no-speech') {
                    this.addTranscript('⚠️ 未检测到语音', 'system');
                } else {
                    this.addTranscript('⚠️ 识别出错: ' + e.error, 'system');
                }
                this._resetRecordingState();
            };

            this.recognition.start();
            this.addTranscript('🎙️ 正在听...', 'system');

        } catch (e) {
            console.error('[Call] 录音失败:', e);
            this.showToast('录音启动失败: ' + e.message, 'error');
            this._resetRecordingState();
        }
    }

    async _handleCallInput() {
        const text = (this.callRecognizedText || '').trim();
        this._resetRecordingState();

        if (!text || text.length < 2) {
            this.addTranscript('⚠️ 请再说一遍', 'system');
            return;
        }

        // 显示用户说的话
        this.addTranscript(text, 'user');
        this.updateCallStatus('thinking');

        try {
            const resp = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: this.userId,
                    message: text,
                    session_id: this.sessionId,
                    voice_mode: true,
                }),
            });

            const data = await resp.json();
            if (!data.success) throw new Error(data.content || '请求失败');

            // 更新 session
            if (data.session_id) {
                this.sessionId = data.session_id;
            }

            // 显示 AI 回复
            const reply = data.content || '嗯…我没想好怎么说';
            this.addTranscript(reply, 'assistant');

            // 用浏览器原生 TTS 朗读
            this._speakText(reply);

        } catch (e) {
            console.error('[Call] 请求失败:', e);
            this.addTranscript('⚠️ 网络出问题了，再试试吧', 'system');
            this.updateCallStatus('listening');
        }
    }

    _speakText(text) {
        this.stopAiSpeaking();

        const synth = window.speechSynthesis;
        if (!synth) {
            this.addTranscript('⚠️ 浏览器不支持语音合成', 'system');
            this.updateCallStatus('listening');
            return;
        }

        // Chrome bug 修复：先 cancel 清掉卡住的队列
        synth.cancel();

        // 找中文女声
        const voices = synth.getVoices();
        let voice = voices.find(v => v.lang === 'zh-CN' && v.name.includes('Xiao'));
        if (!voice) voice = voices.find(v => v.lang === 'zh-CN');
        if (!voice) voice = voices.find(v => v.lang.startsWith('zh'));

        const utterance = new SpeechSynthesisUtterance(text);
        if (voice) utterance.voice = voice;
        utterance.rate = 1.05;
        utterance.pitch = 1.6;
        utterance.volume = 1.0;
        utterance.lang = 'zh-CN';

        // Chrome bug 修复：保持引用防止 GC
        this.currentUtterance = utterance;
        this.isAiSpeaking = true;
        this.updateCallStatus('speaking');
        document.getElementById('callWave').classList.add('speaking');

        const done = () => {
            this.isAiSpeaking = false;
            this.currentUtterance = null;
            this.updateCallStatus('listening');
            document.getElementById('callWave').classList.remove('speaking');
        };

        utterance.onend = done;
        utterance.onerror = (e) => {
            console.error('[Call] TTS error:', e.error);
            // Chrome 有时报 "canceled" 但实际正常
            if (e.error !== 'canceled') {
                this.addTranscript('⚠️ 语音播放出错: ' + e.error, 'system');
            }
            done();
        };

        // Chrome bug 修复：延迟一小段再 speak，防止被浏览器忽略
        setTimeout(() => {
            synth.speak(utterance);
            // Chrome bug 修复：speak 后立即 resume
            synth.resume();
        }, 100);
    }

    _resetRecordingState() {
        this.isCallRecording = false;
        document.getElementById('speakBtn').classList.remove('recording');
        document.getElementById('callWave').classList.remove('active');
    }

    async stopSpeaking() {
        if (!this.isCallRecording) return;

        // 停止语音识别
        if (this.recognition) {
            try { this.recognition.stop(); } catch (e) { /* 忽略 */ }
        }

        // _handleCallInput 会在 recognition.onend 中触发
    }

    interruptAi() {
        if (this.isAiSpeaking) {
            window.speechSynthesis?.cancel();
            this.isAiSpeaking = false;
            this.currentUtterance = null;
            this.updateCallStatus('listening');
            document.getElementById('callWave').classList.remove('speaking');
            this.addTranscript('⏹ 已打断', 'system');
        }
    }

    stopAiSpeaking() {
        if (this.currentUtterance) {
            window.speechSynthesis?.cancel();
            this.currentUtterance = null;
        }
        this.isAiSpeaking = false;
        document.getElementById('callWave').classList.remove('speaking');
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        const muteBtn = document.getElementById('muteBtn');
        if (this.isMuted) {
            muteBtn.classList.add('muted');
            muteBtn.textContent = '🔇';
            // 静音：打断当前播放
            this.stopAiSpeaking();
            this.showToast('🔇 已静音', 'warning');
        } else {
            muteBtn.classList.remove('muted');
            muteBtn.textContent = '🎙️';
            this.showToast('🎙️ 已开启', 'success');
        }
    }

    updateCallStatus(state) {
        const statusEl = document.getElementById('callStatus');
        const avatarWrapper = document.getElementById('callAvatar').parentElement;

        // 清除所有状态类
        avatarWrapper.classList.remove('speaking', 'thinking');
        statusEl.classList.remove('state-listening', 'state-thinking', 'state-speaking');

        switch (state) {
            case 'listening':
                statusEl.textContent = '正在听...';
                statusEl.classList.add('state-listening');
                break;
            case 'thinking':
                statusEl.textContent = '思考中...';
                statusEl.classList.add('state-thinking');
                avatarWrapper.classList.add('thinking');
                break;
            case 'speaking':
                statusEl.textContent = '正在说...';
                statusEl.classList.add('state-speaking');
                avatarWrapper.classList.add('speaking');
                break;
            default:
                statusEl.textContent = state;
        }
    }

    addTranscript(text, role) {
        const container = document.getElementById('callTranscript');
        const div = document.createElement('div');
        div.className = 'transcript-item ' + (role || 'system');
        div.textContent = text;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    updateCallTimer() {
        if (!this.callStartTime) return;
        const elapsed = Math.floor((Date.now() - this.callStartTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        document.getElementById('callTimer').textContent = `${mins}:${secs}`;
    }

    toggleSidebar() {
        document.getElementById('sidebar').classList.toggle('open');
    }
}

// ==================== 全局函数 ====================
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new LifeAgent();
});

function sendMessage() { app.sendMessage(); }
function sendQuick(text) { app.sendMessage(text); }
function newChat() { app.newChat(); }
function toggleSidebar() { app.toggleSidebar(); }
function toggleVoice() { app.toggleVoice(); }
function showPanel(id) { app.showPanel(id); }
function closePanel(id) { app.closePanel(id); }
function generateImage() { app.generateImage(); }

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}
