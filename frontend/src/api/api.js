const API_BASE_URL = 'http://localhost:8000'; // FastAPI URL

export const checkAPIHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};

export const getAIMessage = async (userQuery, sessionId = null) => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: userQuery,
        session_id: sessionId
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      role: data.role,
      content: data.content,
      sessionId: data.session_id,
      metadata: data.metadata || {}
    };
  } catch (error) {
    console.error('Error calling AI API:', error);
    return {
      role: "assistant",
      content: "System Error: I cannot connect to the Hospital Administrator agent. Please check the backend console.",
      error: true
    };
  }
};