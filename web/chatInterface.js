// RLHF Chat Interface
// Professional chat interface for the RLHF system

class RLHFChatInterface {
    constructor() {
        this.messageContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.init();
    }

    init() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Display user message
        this.appendMessage('user', message);
        this.messageInput.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // SECURITY FIX: Use backend proxy endpoint instead of direct API calls
            // This prevents API key exposure in client-side code
            const response = await fetch('/api/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // No API key needed - backend handles authentication
                },
                body: JSON.stringify({
                    messages: [
                        { role: 'system', content: 'You are an AI assistant that provides helpful, accurate, and professional responses.' },
                        { role: 'user', content: message }
                    ],
                    max_tokens: 500,
                    temperature: 0.7
                })
            });

            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            if (data.success && data.completion) {
                this.appendMessage('assistant', data.completion);
            } else {
                this.appendMessage('assistant', 'Sorry, I encountered an error processing your request.');
            }

            // Add active state indicator
            document.body.classList.add('rlhf-system-active');
        } catch (error) {
            this.hideTypingIndicator();
            this.appendMessage('assistant', 'Connection error. Please check your API configuration.');
            document.body.classList.remove('rlhf-system-active');
        }
    }

    appendMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const roleLabel = role === 'user' ? 'User' : 'Assistant';
        messageDiv.innerHTML = `
            <div class="message-role">${roleLabel}</div>
            <div class="message-content">${this.formatMessage(content)}</div>
            <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
        `;
        
        this.messageContainer.appendChild(messageDiv);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    formatMessage(content) {
        // Basic markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <div class="message assistant">
                <div class="message-role">Assistant</div>
                <div class="message-content">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        
        this.messageContainer.appendChild(indicator);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
}

// Initialize chat interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    const chatInterface = new RLHFChatInterface();
    console.log('RLHF Chat Interface initialized');
});
