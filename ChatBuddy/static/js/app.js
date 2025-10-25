/**
 * Senior Citizens Chat Buddy - Frontend JavaScript
 * Handles chat interface interactions and API communication
 */

class ChatBuddy {
    constructor() {
        this.isLoading = false;
        this.messageHistory = [];
        this.maxRetries = 3;
        this.retryDelay = 1000;
        
        this.initializeElements();
        this.bindEvents();
        this.loadConversationHistory();
        this.setupAccessibility();
        
        console.log('✅ Chat Buddy initialized successfully');
    }
    
    initializeElements() {
        // Main elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.charCount = document.getElementById('charCount');
        
        // Control buttons
        this.clearBtn = document.getElementById('clearBtn');
        this.helpBtn = document.getElementById('helpBtn');
        this.closeHelpBtn = document.getElementById('closeHelpBtn');
        
        // Modals and overlays
        this.helpModal = document.getElementById('helpModal');
        this.errorToast = document.getElementById('errorToast');
        this.errorMessage = document.getElementById('errorMessage');
        this.closeErrorBtn = document.getElementById('closeErrorBtn');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Quick action buttons
        this.quickBtns = document.querySelectorAll('.quick-btn');
    }
    
    bindEvents() {
        // Send message events
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Character counter
        this.messageInput.addEventListener('input', () => this.updateCharCounter());
        
        // Control buttons
        this.clearBtn.addEventListener('click', () => this.clearConversation());
        this.helpBtn.addEventListener('click', () => this.showHelp());
        this.closeHelpBtn.addEventListener('click', () => this.hideHelp());
        
        // Error handling
        this.closeErrorBtn.addEventListener('click', () => this.hideError());
        
        // Quick actions
        this.quickBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const message = btn.getAttribute('data-message');
                this.messageInput.value = message;
                this.updateCharCounter();
                this.sendMessage();
            });
        });
        
        // Modal click outside to close
        this.helpModal.addEventListener('click', (e) => {
            if (e.target === this.helpModal) {
                this.hideHelp();
            }
        });
        
        // Auto-hide error toast
        this.errorToast.addEventListener('click', () => this.hideError());
        
        // Focus management
        this.messageInput.addEventListener('focus', () => {
            this.messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    }
    
    setupAccessibility() {
        // ARIA live region for screen readers
        this.messagesContainer.setAttribute('aria-live', 'polite');
        this.messagesContainer.setAttribute('aria-label', 'Chat conversation');
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            // Escape key closes modals
            if (e.key === 'Escape') {
                this.hideHelp();
                this.hideError();
            }
            
            // Focus management
            if (e.key === 'Tab') {
                this.handleTabNavigation(e);
            }
        });
        
        // Announce new messages to screen readers
        this.announceToScreenReader = (message) => {
            const announcement = document.createElement('div');
            announcement.setAttribute('aria-live', 'assertive');
            announcement.setAttribute('aria-atomic', 'true');
            announcement.className = 'sr-only';
            announcement.textContent = `New message: ${message}`;
            document.body.appendChild(announcement);
            
            setTimeout(() => {
                document.body.removeChild(announcement);
            }, 1000);
        };
    }
    
    handleTabNavigation(e) {
        // Ensure proper tab order
        const focusableElements = [
            this.messageInput,
            this.sendBtn,
            this.clearBtn,
            this.helpBtn,
            ...this.quickBtns
        ];
        
        const currentIndex = focusableElements.indexOf(document.activeElement);
        
        if (e.shiftKey && currentIndex === 0) {
            // Shift+Tab from first element - focus last
            e.preventDefault();
            focusableElements[focusableElements.length - 1].focus();
        } else if (!e.shiftKey && currentIndex === focusableElements.length - 1) {
            // Tab from last element - focus first
            e.preventDefault();
            focusableElements[0].focus();
        }
    }
    
    updateCharCounter() {
        const count = this.messageInput.value.length;
        this.charCount.textContent = count;
        
        // Visual feedback for character limit
        if (count > 450) {
            this.charCount.style.color = '#e74c3c';
        } else if (count > 400) {
            this.charCount.style.color = '#f39c12';
        } else {
            this.charCount.style.color = '#7f8c8d';
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) {
            this.showError('Please enter a message');
            this.messageInput.focus();
            return;
        }
        
        if (message.length > 500) {
            this.showError('Message is too long. Please keep it under 500 characters.');
            this.messageInput.focus();
            return;
        }
        
        if (this.isLoading) {
            return; // Prevent multiple simultaneous requests
        }
        
        try {
            this.isLoading = true;
            this.setLoadingState(true);
            
            // Add user message to UI immediately
            this.addMessageToUI('user', message);
            this.messageInput.value = '';
            this.updateCharCounter();
            
            // Show typing indicator
            this.showTypingIndicator();
            
            // Send to backend
            const response = await this.sendToBackend(message);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            if (response.success) {
                // Add bot response to UI
                this.addMessageToUI('bot', response.bot_message.content);
                
                // Announce to screen reader
                this.announceToScreenReader(response.bot_message.content);
                
                // Save to history
                this.messageHistory.push(response.user_message);
                this.messageHistory.push(response.bot_message);
            } else {
                throw new Error(response.error || 'Failed to get response');
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.showError('Sorry, I\'m having trouble connecting. Please try again.');
            
            // Remove the user message if sending failed
            this.removeLastUserMessage();
        } finally {
            this.isLoading = false;
            this.setLoadingState(false);
            this.messageInput.focus();
        }
    }
    
    async sendToBackend(message, retryCount = 0) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Unknown error');
            }
            
            return data;
            
        } catch (error) {
            console.error(`Attempt ${retryCount + 1} failed:`, error);
            
            if (retryCount < this.maxRetries - 1) {
                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * (retryCount + 1)));
                return this.sendToBackend(message, retryCount + 1);
            } else {
                throw error;
            }
        }
    }
    
    addMessageToUI(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        // Handle line breaks in content
        const formattedContent = content.replace(/\n/g, '<br>');
        messageText.innerHTML = formattedContent;
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = this.formatTime(new Date());
        
        messageContent.appendChild(messageText);
        messageContent.appendChild(messageTime);
        messageDiv.appendChild(messageContent);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    removeLastUserMessage() {
        const messages = this.messagesContainer.querySelectorAll('.user-message');
        if (messages.length > 0) {
            const lastMessage = messages[messages.length - 1];
            lastMessage.remove();
        }
    }
    
    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
    
    setLoadingState(loading) {
        this.sendBtn.disabled = loading;
        this.messageInput.disabled = loading;
        
        if (loading) {
            this.sendBtn.innerHTML = '<span class="send-icon">⏳</span> Sending...';
        } else {
            this.sendBtn.innerHTML = '<span class="send-icon">📤</span> Send';
        }
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }
    
    async clearConversation() {
        if (!confirm('Are you sure you want to clear the conversation? This cannot be undone.')) {
            return;
        }
        
        try {
            this.setLoadingState(true);
            
            const response = await fetch('/api/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (response.ok) {
                // Clear UI
                const messages = this.messagesContainer.querySelectorAll('.message:not(.welcome-message)');
                messages.forEach(msg => msg.remove());
                
                // Clear history
                this.messageHistory = [];
                
                // Show success message
                this.showSuccess('Conversation cleared successfully');
                
                // Focus back to input
                this.messageInput.focus();
            } else {
                throw new Error('Failed to clear conversation');
            }
            
        } catch (error) {
            console.error('Error clearing conversation:', error);
            this.showError('Failed to clear conversation. Please try again.');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    async loadConversationHistory() {
        try {
            const response = await fetch('/api/history');
            
            if (response.ok) {
                const data = await response.json();
                
                if (data.success && data.messages.length > 0) {
                    // Clear existing messages except welcome
                    const messages = this.messagesContainer.querySelectorAll('.message:not(.welcome-message)');
                    messages.forEach(msg => msg.remove());
                    
                    // Add history messages
                    data.messages.forEach(msg => {
                        this.addMessageToUI(msg.type, msg.content);
                        this.messageHistory.push(msg);
                    });
                    
                    this.scrollToBottom();
                }
            }
        } catch (error) {
            console.error('Error loading conversation history:', error);
            // Don't show error to user for history loading
        }
    }
    
    showHelp() {
        this.helpModal.style.display = 'flex';
        this.closeHelpBtn.focus();
        
        // Prevent body scroll
        document.body.style.overflow = 'hidden';
    }
    
    hideHelp() {
        this.helpModal.style.display = 'none';
        document.body.style.overflow = '';
        this.helpBtn.focus();
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorToast.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
        
        // Announce to screen reader
        this.announceToScreenReader(`Error: ${message}`);
    }
    
    hideError() {
        this.errorToast.style.display = 'none';
    }
    
    showSuccess(message) {
        // Create temporary success toast
        const successToast = document.createElement('div');
        successToast.className = 'toast';
        successToast.style.background = '#27ae60';
        successToast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">✅</span>
                <span class="toast-message">${message}</span>
            </div>
        `;
        
        document.body.appendChild(successToast);
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (document.body.contains(successToast)) {
                document.body.removeChild(successToast);
            }
        }, 3000);
    }
    
    showLoadingOverlay() {
        this.loadingOverlay.style.display = 'flex';
    }
    
    hideLoadingOverlay() {
        this.loadingOverlay.style.display = 'none';
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.chatBuddy = new ChatBuddy();
        
        // Health check on load
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.agent_available) {
                    console.log('✅ Chat agent is ready');
                } else {
                    console.warn('⚠️ Chat agent is not available');
                    window.chatBuddy.showError('Chat agent is not available. Please refresh the page.');
                }
            })
            .catch(error => {
                console.error('Health check failed:', error);
                window.chatBuddy.showError('Unable to connect to chat service. Please check your connection.');
            });
            
    } catch (error) {
        console.error('Failed to initialize Chat Buddy:', error);
        alert('Failed to initialize the chat interface. Please refresh the page.');
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.chatBuddy) {
        // Reload conversation history when page becomes visible
        window.chatBuddy.loadConversationHistory();
    }
});

// Handle online/offline status
window.addEventListener('online', () => {
    if (window.chatBuddy) {
        window.chatBuddy.showSuccess('Connection restored');
    }
});

window.addEventListener('offline', () => {
    if (window.chatBuddy) {
        window.chatBuddy.showError('You are offline. Please check your internet connection.');
    }
});

// Prevent form submission on Enter in input
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.tagName === 'INPUT' && e.target.type === 'text') {
        e.preventDefault();
    }
});

// Add CSS for screen reader only content
const style = document.createElement('style');
style.textContent = `
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
`;
document.head.appendChild(style);
