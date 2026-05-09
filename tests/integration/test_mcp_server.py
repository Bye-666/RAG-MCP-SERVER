"""
Integration tests for MCP Server stdio communication.

Tests verify:
- JSON-RPC 2.0 protocol compliance
- stdout only contains MCP messages
- stderr contains logs
- initialize handshake works
"""

import json
import subprocess
import sys
from pathlib import Path


def test_server_initialize():
    """Test server can complete initialize handshake."""
    # Prepare initialize request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    # Start server process
    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send request
    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n", timeout=5)

    # Verify stdout contains valid JSON-RPC response
    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 1, "Should have at least one response line"

    response = json.loads(lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert "capabilities" in response["result"]
    assert "serverInfo" in response["result"]

    # Verify stderr contains logs (not empty)
    assert len(stderr) > 0, "stderr should contain logs"
    assert "initialize" in stderr.lower() or "server" in stderr.lower()


def test_server_stdout_not_polluted():
    """Test stdout only contains JSON-RPC messages, no log pollution."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }

    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n", timeout=5)

    # Every line in stdout should be valid JSON
    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    for line in lines:
        data = json.loads(line)  # Should not raise
        assert "jsonrpc" in data, "Every stdout line should be JSON-RPC message"

    # stderr should have logs
    assert len(stderr) > 0


def test_server_invalid_request():
    """Test server handles invalid JSON-RPC request."""
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "unknown_method",
        "params": {}
    }

    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n", timeout=5)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 1

    response = json.loads(lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 3
    assert "error" in response
    assert response["error"]["code"] == -32601  # Method not found


def test_server_malformed_json():
    """Test server handles malformed JSON input."""
    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input="not valid json\n", timeout=5)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 1

    response = json.loads(lines[0])
    assert response["jsonrpc"] == "2.0"
    assert "error" in response
    assert response["error"]["code"] == -32700  # Parse error


def test_server_multiple_requests():
    """Test server can handle multiple sequential requests."""
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        },
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "initialized",
            "params": {}
        }
    ]

    input_data = "\n".join(json.dumps(req) for req in requests) + "\n"

    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=input_data, timeout=5)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 2, "Should have responses for both requests"

    # First response: initialize
    resp1 = json.loads(lines[0])
    assert resp1["id"] == 10
    assert "result" in resp1

    # Second response: initialized (notification, may not have response)
    # Or error if not initialized yet
    resp2 = json.loads(lines[1])
    assert resp2["jsonrpc"] == "2.0"
