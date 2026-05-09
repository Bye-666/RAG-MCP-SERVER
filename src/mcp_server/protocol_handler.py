"""
Protocol Handler for MCP Server.

Handles JSON-RPC 2.0 protocol parsing and tool routing.
"""

from typing import Any, Dict, Callable, List
import logging


logger = logging.getLogger(__name__)


class JSONRPCError(Exception):
    """JSON-RPC error with code and message."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ProtocolHandler:
    """
    Handles MCP protocol operations.

    Manages tool registration and routing for JSON-RPC 2.0 requests.
    """

    def __init__(self):
        self.protocol_version = "2024-11-05"
        self.server_info = {
            "name": "rag-mcp-server",
            "version": "0.1.0"
        }
        self.tools = {}  # name -> tool definition

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ):
        """
        Register a tool with the protocol handler.

        Args:
            name: Tool name (unique identifier)
            description: Human-readable description
            input_schema: JSON Schema for tool input
            handler: Callable that implements the tool
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler
        }
        logger.info(f"Registered tool: {name}")

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle initialize request.

        Args:
            params: Initialize parameters (clientInfo, etc.)

        Returns:
            Initialize result with serverInfo and capabilities
        """
        logger.info(f"Initialize request: {params}")

        return {
            "protocolVersion": self.protocol_version,
            "serverInfo": self.server_info,
            "capabilities": {
                "tools": {}
            }
        }

    def handle_tools_list(self) -> Dict[str, Any]:
        """
        Handle tools/list request.

        Returns:
            List of registered tool schemas
        """
        tools_list = []
        for tool_def in self.tools.values():
            tools_list.append({
                "name": tool_def["name"],
                "description": tool_def["description"],
                "inputSchema": tool_def["inputSchema"]
            })

        return {"tools": tools_list}

    def handle_tools_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Handle tools/call request.

        Routes to the appropriate tool handler and executes it.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            JSONRPCError: With appropriate error code
        """
        # Check if tool exists
        if name not in self.tools:
            raise JSONRPCError(-32601, f"Tool not found: {name}")

        tool_def = self.tools[name]
        handler = tool_def["handler"]

        try:
            # Call tool handler
            result = handler(**arguments)
            return result

        except TypeError as e:
            # Invalid parameters (wrong argument types/names)
            logger.warning(f"Invalid parameters for tool {name}: {e}")
            raise JSONRPCError(-32602, "Invalid params")

        except Exception as e:
            # Internal error - don't leak details
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            raise JSONRPCError(-32603, "Internal error")
