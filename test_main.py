from fastapi.testclient import TestClient
from unittest.mock import patch
from langchain_core.messages import AIMessage, ToolMessage
from main import app

# This creates a fake browser/client to send requests to our API
client = TestClient(app)

# @patch intercepts the invoke method of our agent inside main.py
@patch("main.agent.invoke")
def test_chat_endpoint_success(mock_agent_invoke):
    # 1. Define the fake data we want the agent to return
    # We simulate a scenario where the agent used a tool, then gave a final answer.
    mock_agent_invoke.return_value = {
        "messages": [
            ToolMessage(
                content="Account Active.", 
                name="check_account_status", 
                tool_call_id="fake_id"
            ),
            AIMessage(
                content="Your account is currently Active."
            )
        ]
    }

    # 2. Send a POST request to our local test client
    response = client.post(
        "/chat",
        json={"query": "What is my account status?", "user_id": "test_user"}
    )

    # 3. Assert (verify) the results
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify the API successfully parsed our mocked AI Message
    assert "Active" in data["answer"]
    
    # Verify the API successfully caught the tool usage
    assert "check_account_status" in data["tools_used"]