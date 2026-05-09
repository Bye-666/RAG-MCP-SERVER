"""
Unit tests for ProtocolHandler.
"""
import pytest
from src.mcp_server.protocol_handler import ProtocolHandler


class TestProtocolHandler:
    """Test ProtocolHandler JSON-RPC 2.0 protocol handling."""

    @pytest.fixture
    def handler(self):
        """Create a ProtocolHandler instance."""
        return ProtocolHandler()

    def test_handle_initialize_returns_server_info(self, handler):
        """Test initialize returns serverInfo and capabilities."""
        result = handler.handle_initialize({})

        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "rag-mcp-server"
        assert "version" in result["serverInfo"]

        assert "capabilities" in result
        assert "tools" in result["capabilities"]

    def test_handle_initialize_with_client_info(self, handler):
        """Test initialize accepts clientInfo parameter."""
        params = {
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
        result = handler.handle_initialize(params)

        assert "serverInfo" in result
        assert "capabilities" in result

    def test_handle_tools_list_empty(self, handler):
        """Test tools/list returns empty array when no tools registered."""
        result = handler.handle_tools_list()

        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) == 0

    def test_handle_tools_list_with_registered_tools(self, handler):
        """Test tools/list returns registered tool schemas."""
        def dummy_tool(query: str) -> dict:
            return {"result": query}

        handler.register_tool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            },
            handler=dummy_tool
        )

        result = handler.handle_tools_list()

        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["name"] == "test_tool"
        assert tool["description"] == "A test tool"
        assert "inputSchema" in tool

    def test_handle_tools_call_success(self, handler):
        """Test tools/call routes to registered tool and returns result."""
        def dummy_tool(query: str) -> dict:
            return {"result": f"processed: {query}"}

        handler.register_tool(
            name="test_tool",
            description="A test tool",
            input_schema={},
            handler=dummy_tool
        )

        result = handler.handle_tools_call("test_tool", {"query": "hello"})

        assert result == {"result": "processed: hello"}

    def test_handle_tools_call_method_not_found(self, handler):
        """Test tools/call returns -32601 for unknown tool."""
        with pytest.raises(Exception) as exc_info:
            handler.handle_tools_call("unknown_tool", {})

        error = exc_info.value
        assert hasattr(error, "code")
        assert error.code == -32601

    def test_handle_tools_call_invalid_params(self, handler):
        """Test tools/call returns -32602 for invalid parameters."""
        def strict_tool(query: str) -> dict:
            if not isinstance(query, str):
                raise TypeError("query must be string")
            return {"result": query}

        handler.register_tool(
            name="strict_tool",
            description="A strict tool",
            input_schema={},
            handler=strict_tool
        )

        with pytest.raises(Exception) as exc_info:
            handler.handle_tools_call("strict_tool", {"query": 123})

        error = exc_info.value
        assert hasattr(error, "code")
        assert error.code == -32602

    def test_handle_tools_call_internal_error(self, handler):
        """Test tools/call returns -32603 for internal exceptions."""
        def failing_tool(query: str) -> dict:
            raise RuntimeError("Internal failure")

        handler.register_tool(
            name="failing_tool",
            description="A failing tool",
            input_schema={},
            handler=failing_tool
        )

        with pytest.raises(Exception) as exc_info:
            handler.handle_tools_call("failing_tool", {"query": "test"})

        error = exc_info.value
        assert hasattr(error, "code")
        assert error.code == -32603
        assert "Internal failure" not in str(error)  # Should not leak stack trace

    def test_register_tool_validates_schema(self, handler):
        """Test register_tool validates input schema."""
        def dummy_tool(query: str) -> dict:
            return {"result": query}

        handler.register_tool(
            name="valid_tool",
            description="Valid tool",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            },
            handler=dummy_tool
        )

        result = handler.handle_tools_list()
        assert len(result["tools"]) == 1

    def test_handle_tools_call_with_no_arguments(self, handler):
        """Test tools/call works with tools that take no arguments."""
        def no_arg_tool() -> dict:
            return {"result": "success"}

        handler.register_tool(
            name="no_arg_tool",
            description="Tool with no args",
            input_schema={"type": "object", "properties": {}},
            handler=no_arg_tool
        )

        result = handler.handle_tools_call("no_arg_tool", {})
        assert result == {"result": "success"}

    def test_multiple_tools_registration(self, handler):
        """Test registering multiple tools."""
        def tool1(x: int) -> dict:
            return {"result": x * 2}

        def tool2(y: str) -> dict:
            return {"result": y.upper()}

        handler.register_tool("tool1", "First tool", {}, tool1)
        handler.register_tool("tool2", "Second tool", {}, tool2)

        result = handler.handle_tools_list()
        assert len(result["tools"]) == 2

        names = [t["name"] for t in result["tools"]]
        assert "tool1" in names
        assert "tool2" in names

    def test_error_does_not_leak_sensitive_info(self, handler):
        """Test internal errors don't leak sensitive information."""
        def sensitive_tool(query: str) -> dict:
            secret = "SECRET_API_KEY_12345"
            raise ValueError(f"Failed with {secret}")

        handler.register_tool(
            name="sensitive_tool",
            description="Tool with sensitive data",
            input_schema={},
            handler=sensitive_tool
        )

        with pytest.raises(Exception) as exc_info:
            handler.handle_tools_call("sensitive_tool", {"query": "test"})

        error = exc_info.value
        error_message = str(error)
        assert "SECRET_API_KEY" not in error_message
        assert error.code == -32603
