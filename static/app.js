/**
 * 生活管家 AI-Agent 前端应用
 * 功能：聊天、会话管理、文件上传、记忆管理
 */
class LifeAgent {
    constructor() {
        this.sessionId = null;
        this.isLoading = false;
        this.apiBase = '/api';
        this.pendingImage = null;  // base64 string
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadSessions();
        this.autoResizeInput();
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
        this.sendBtn.disabled = true;
        this.inputEl.value = '';
        this.inputEl.style.height = 'auto';

        // 清除欢迎界面
        this.clearWelcome();

        // 显示用户消息（附带图片预览）
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

        // 显示加载状态
        const loadingId = this.showLoading();

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
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            // 保存 session_id
            if (data.session_id) {
                this.sessionId = data.session_id;
                this.loadSessions();
            }

            // 移除加载状态，显示 AI 回复
            this.removeLoading(loadingId);

            // 如果需要人机确认（危险操作）
            if (data.need_confirm && data.pending_id) {
                this.showConfirmDialog(data.pending_id, data.confirm_preview || data.content);
            }

            this.appendMessage('ai', data.content, elapsed, data.tool_used);

        } catch (error) {
            this.removeLoading(loadingId);
            this.appendMessage('ai', `⚠️ 请求失败: ${error.message}\n请检查网络连接或稍后重试。`);
        } finally {
            this.isLoading = false;
            this.sendBtn.disabled = false;
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

        // system 消息不需要头像
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

        // 简单的 Markdown 渲染
        let html = this.renderMarkdown(content);

        // 工具标记
        if (toolUsed) {
            const toolNames = {
                calculator: '🔢 计算器',
                get_current_time: '🕐 时间查询',
                save_memory: '🧠 记忆保存',
                search_documents: '📄 文档检索',
            };
            html = `<div style="font-size:11px;color:var(--text-light);margin-bottom:6px;">使用了 ${toolNames[toolUsed] || toolUsed}</div>` + html;
        }

        // 耗时标记
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
        bubble.innerHTML = `
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;

        div.appendChild(avatar);
        div.appendChild(bubble);
        this.messagesEl.appendChild(div);
        this.scrollToBottom();
        return id;
    }

    removeLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    renderMarkdown(text) {
        if (!text) return '';
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

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
        } catch (e) {
            // 静默失败
        }
    }

    renderSessions(sessions) {
        this.sessionListEl.innerHTML = '';
        sessions.forEach(s => {
            const div = document.createElement('div');
            div.className = 'session-item' + (s.session_id === this.sessionId ? ' active' : '');
            const date = new Date(s.updated_at).toLocaleString('zh-CN', {
                month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            div.textContent = `💬 对话 ${date}`;
            div.onclick = () => this.switchSession(s.session_id);
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

        // 重置 input 以允许重复上传同一文件
        event.target.value = '';
    }

    // ==================== 工具方法 ====================

    showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
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
        // 关闭弹窗
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

        // 确认执行
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

    autoResizeInput() {
        this.inputEl.addEventListener('input', () => {
            this.inputEl.style.height = 'auto';
            this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + 'px';
        });
    }

    // ==================== 图片上传 ====================

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type.startsWith('image/')) {
            // 图片 → base64 预览
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
            // 文本文件 → 直接读取内容作为消息
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

    toggleSidebar() {
        document.getElementById('sidebar').classList.toggle('open');
    }
}

// ==================== 全局函数 ====================
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new LifeAgent();
});

function sendMessage() {
    app.sendMessage();
}

function sendQuick(text) {
    app.sendMessage(text);
}

function newChat() {
    app.newChat();
}

function toggleSidebar() {
    app.toggleSidebar();
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}
