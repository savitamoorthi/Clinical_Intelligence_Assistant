import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; // Run: npm install remark-gfm
import { FaUserMd, FaUser } from 'react-icons/fa'; // Icons
import './ChatWindow.css';

function ChatWindow({ messages, isLoading }) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, isLoading]);

  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <div key={index} className={`message-row ${msg.role}`}>
          
          {/* Avatar for Assistant */}
          {msg.role === 'assistant' && (
            <div className="avatar assistant">
              <FaUserMd />
            </div>
          )}

          <div className="message-content">
            {msg.role === 'assistant' ? (
              // remarkGfm allows for nice table rendering
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            ) : (
              <p>{msg.content}</p>
            )}
          </div>

          {/* Avatar for User (Optional - currently hidden via CSS) */}
          {msg.role === 'user' && (
            <div className="avatar user">
              <FaUser />
            </div>
          )}
        </div>
      ))}
      
      {isLoading && (
        <div className="message-row assistant">
          <div className="avatar assistant">
            <FaUserMd />
          </div>
          <div className="message-content">
            <div className="typing-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default ChatWindow;