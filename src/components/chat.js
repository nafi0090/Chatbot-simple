import axios from 'axios';
import React, { useState } from 'react';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    try {
      const response = await axios.post('http://localhost:5000/chat', { message: input }, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      });
  
      const botResponse = response.data.botResponse;
      const newMessages = [...messages, { text: input, type: 'user' }, { text: botResponse, type: 'bot' }];
      setMessages(newMessages);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  
    setInput('');
  };
  

  return (
    <div>
      <div>
        {messages.map((message, index) => (
          <div key={index} className={message.type}>
            {message.type === 'user' ? 'User' : 'Bot'}: {message.text}
          </div>
        ))}
      </div>
      <div>
        <input type="text" value={input} onChange={(e) => setInput(e.target.value)} />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

export default Chat;
