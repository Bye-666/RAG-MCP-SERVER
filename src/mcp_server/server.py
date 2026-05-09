"""
MCP Server entry point with stdio transport.

Implements JSON-RPC 2.0 protocol over stdin/stdout:
- Reads JSON-RPC requests from stdin (one per line)
- Writes JSON-RPC responses to stdout (one per line)
- Logs to stderr (never pollute stdout)
"""

import json
import logging
import sys
from typing import Any, Dict, Optional


# Configure logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server implementing JSON-RPC 2.0 protocol."""

    def __init__(self):
        self.initialized = False
        self.protocol_version = "2024-11-05"
        self.server_info = {
            "name": "rag-mcp-server",
            "version": "0.1.0"
        }
        self.capabilities = {
            "tools": {}
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a JSON-RPC 2.0 request.

        Args:
            request: JSON-RPC request object

        Returns:
            JSON-RPC response object
        """
        # Validate JSON-RPC version
        if request.get("jsonrpc") != "2.0":
            return self._error_response(
                request.get("id"),
                -32600,
                "Invalid Request: jsonrpc must be '2.0'"
            )

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        logger.info(f"Handling request: method={method}, id={request_id}")

        # Route to handler
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "initialized":
            return self._handle_initialized(request_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        else:
            return self._error_response(
                request_id,
                -32601,
                f"Method not found: {method}"
            )

    def _handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        logger.info(f"Initialize request: {params}")

        # Validate protocol version
        client_version = params.get("protocolVersion")
        if client_version != self.protocol_version:
            logger.warning(f"Protocol version mismatch: client={client_version}, server={self.protocol_version}")

        self.initialized = True

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": self.protocol_version,
                "capabilities": self.capabilities,
                "serverInfo": self.server_info
            }
        }

    def _handle_initialized(self, request_id: Optional[int], params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialized notification."""
        logger.info("Client sent initialized notification")

        # initialized is a notification, may not have id
        if request_id is None:
            # No response for notifications
            return None

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }

    def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools/list request."""
        if not self.initialized:
            return self._error_response(
                request_id,
                -32002,
                "Server not initialized"
            )

        # TODO: Return registered tools (E2 will implement tool registry)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": []
            }
        }

    def _handle_tools_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        if not self.initialized:
            return self._error_response(
                request_id,
                -32002,
                "Server not initialized"
            )

        tool_name = params.get("name")
        if not tool_name:
            return self._error_response(
                request_id,
                -32602,
                "Invalid params: 'name' is required"
            )

        # TODO: Route to tool implementation (E3+ will implement tools)
        return self._error_response(
            request_id,
            -32601,
            f"Tool not found: {tool_name}"
        )

    def _error_response(self, request_id: Optional[int], code: int, message: str) -> Dict[str, Any]:
        """Create JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    def run(self):
        """
        Main server loop: read from stdin, write to stdout.

        Reads JSON-RPC requests line by line from stdin.
        Writes JSON-RPC responses line by line to stdout.
        """
        logger.info("MCP Server starting...")
        logger.info(f"Server info: {self.server_info}")
        logger.info("Listening on stdin for JSON-RPC requests...")

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON-RPC request
                    request = json.loads(line)
                    logger.debug(f"Received request: {request}")

                    # Handle request
                    response = self.handle_request(request)

                    # Write response to stdout (if not None)
                    if response is not None:
                        response_json = json.dumps(response)
                        print(response_json, flush=True)
                        logger.debug(f"Sent response: {response}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    error_response = self._error_response(
                        None,
                        -32700,
                        f"Parse error: {str(e)}"
                    )
                    print(json.dumps(error_response), flush=True)

                except Exception as e:
                    logger.error(f"Internal error: {e}", exc_info=True)
                    error_response = self._error_response(
                        None,
                        -32603,
                        "Internal error"
                    )
                    print(json.dumps(error_response), flush=True)

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            logger.info("MCP Server shutting down")


def main():
    """Entry point."""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
