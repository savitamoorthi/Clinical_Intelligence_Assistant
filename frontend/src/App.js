import React, { useState, useEffect } from 'react';
import { FaPaperPlane } from 'react-icons/fa';
import ChatWindow from './components/ChatWindow';
import { getAIMessage, checkAPIHealth } from './api/api';
import './App.css';

// --- CHANGE: Importing PNG ---
import logo from './assets/logo.png';
// -----------------------------

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      const healthy = await checkAPIHealth();
      setIsConnected(healthy);
      if (healthy) {
        setMessages([{
          role: 'assistant',
          content: "Welcome to the **UChicago Medicine Clinical Intelligence Assistant**. \n\nI can assist you with:\n* **EHR Data:** Patient demographics, admissions, and vitals.\n* **Radiology:** Searching and summarizing clinical reports.\n\nHow can I help you today?",
          timestamp: new Date().toISOString()
        }]);
      } else {
        setMessages([{
          role: 'assistant',
          content: "⚠️ **System Offline.**\n\nPlease ensure the GenBI Backend Server is running (`python backend_server.py`).",
          timestamp: new Date().toISOString()
        }]);
      }
    };
    checkHealth();
  }, []);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    const response = await getAIMessage(input, sessionId);

    if (response.sessionId && !sessionId) {
      setSessionId(response.sessionId);
    }

    const aiMsg = {
      role: 'assistant',
      content: response.content,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, aiMsg]);
    setIsLoading(false);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        {/* Left: Logo only */}
        <div className="logo-area">
          <img src={logo} alt="UChicago Medicine" className="logo-image" />
          <div className="divider"></div>
          {/* Title removed from here */}
        </div>

        {/* Center: Title is now a direct child of header */}
        <span className="app-title">Clinical Intelligence Assistant</span>

        {/* Right: Status */}
        <div className={`status-badge ${isConnected ? 'online' : 'offline'}`}>
          {isConnected ? 'System Online' : 'Offline'}
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="main-content">
        <ChatWindow messages={messages} isLoading={isLoading} />

        {/* Floating Input Area (Gemini Style) */}
        <div className="input-area-wrapper">
          <form onSubmit={handleSend} className="input-container">
            <input
              className="input-box"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about patient records or radiology findings..."
              disabled={isLoading || !isConnected}
            />
            <button
              type="submit"
              className="send-button"
              disabled={isLoading || !isConnected || !input.trim()}
            >
              <FaPaperPlane size={15} style={{ marginLeft: '2px' }} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;