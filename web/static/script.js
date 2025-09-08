// web/static/script.js (With "New Chat" functionality)
document.addEventListener('DOMContentLoaded', () => {
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const messagesContainer = document.getElementById('messages-container');
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatInterface = document.getElementById('chat-interface');
    const newChatButton = document.getElementById('new-chat-button'); 
    const API_URL = 'http://127.0.0.1:8003/ask';
    const NEW_CHAT_URL = 'http://127.0.0.1:8003/new-chat'; 

    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = messageInput.value.trim();
        if (!question) return;

        if (!welcomeScreen.classList.contains('hidden')) {
            welcomeScreen.classList.add('hidden');
            chatInterface.classList.remove('hidden');
        }

        appendMessage(question, 'user');
        messageInput.value = '';
        
        const thinkingBubble = appendMessage('', 'ai');
        const thinkingIndicator = showThinkingIndicator(thinkingBubble);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: question }),
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            
            if (thinkingIndicator.parentNode) {
                thinkingIndicator.parentNode.removeChild(thinkingIndicator);
            }
            if (data.answer && data.answer.trim() !== '') {
                typewriterEffect(thinkingBubble, data.answer);
            } else {
                thinkingBubble.innerHTML = "ขออภัยครับ ผมไม่สามารถหาคำตอบที่เหมาะสมสำหรับคำถามนี้ได้ในขณะนี้";
            }
            
            typewriterEffect(thinkingBubble, data.answer);

        } catch (error) {
            console.error('Error fetching AI response:', error);
            if (thinkingIndicator.parentNode) thinkingIndicator.parentNode.removeChild(thinkingIndicator);
            thinkingBubble.innerHTML = 'Sorry, something went wrong.';
        }
    });

    newChatButton.addEventListener('click', async () => {
        console.log("New chat button clicked");
        
        try {
            await fetch(NEW_CHAT_URL, { method: 'POST' });
        } catch (error) {
            console.error("Failed to clear chat history on server:", error);
        }
        
        messagesContainer.innerHTML = '';
        
        welcomeScreen.classList.remove('hidden');
        chatInterface.classList.add('hidden');
    });

    function typewriterEffect(element, text) {
        let i = 0;
        element.textContent = '';
        function typing() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                setTimeout(typing, 20);
            } else {
                element.innerHTML = marked.parse(text);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }
        typing();
    }

    function appendMessage(content, type) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `w-full flex mb-4 ${type === 'user' ? 'justify-end' : 'justify-start'}`;

        const messageBubble = document.createElement('div');
        messageBubble.className = `max-w-xl p-3 rounded-lg ${type === 'user' ? 'bg-blue-700 text-white' : 'bg-gray-700 text-gray-200'}`;
        messageBubble.textContent = content;

        messageWrapper.appendChild(messageBubble);
        messagesContainer.appendChild(messageWrapper);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageBubble; 
    }

    function showThinkingIndicator(bubble) {
        bubble.innerHTML = `<div class="thinking-indicator"><span>.</span><span>.</span><span>.</span></div>`;
        return bubble.firstChild;
    }
});