"""
MCP 服务器入口点，使用 stdio 传输。

通过 stdin/stdout 实现 JSON-RPC 2.0 协议：
- 从 stdin 读取 JSON-RPC 请求（每行一个）
- 向 stdout 写入 JSON-RPC 响应（每行一个）
- 日志输出到 stderr（永远不污染 stdout）
"""

import json
import logging
import sys
from typing import Any, Dict, Optional


# 配置日志仅输出到 stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class MCPServer:
    """MCP 服务器，实现 JSON-RPC 2.0 协议。"""

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
        处理 JSON-RPC 2.0 请求。

        Args:
            request: JSON-RPC 请求对象

        Returns:
            JSON-RPC 响应对象
        """
        # 验证 JSON-RPC 版本
        if request.get("jsonrpc") != "2.0":
            return self._error_response(
                request.get("id"),
                -32600,
                "无效请求：jsonrpc 必须为 '2.0'"
            )

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        logger.info(f"Handling request: method={method}, id={request_id}")

        # 路由到处理器
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
                f"方法未找到：{method}"
            )

    def _handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 initialize 请求。"""
        logger.info(f"Initialize request: {params}")

        # 验证协议版本
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
        """处理 initialized 通知。"""
        logger.info("Client sent initialized notification")

        # initialized 是通知，可能没有 id
        if request_id is None:
            # 通知不需要响应
            return None

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }

    def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """处理 tools/list 请求。"""
        if not self.initialized:
            return self._error_response(
                request_id,
                -32002,
                "服务器未初始化"
            )

        # TODO: 返回已注册的工具（E2 将实现工具注册表）
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": []
            }
        }

    def _handle_tools_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 tools/call 请求。"""
        if not self.initialized:
            return self._error_response(
                request_id,
                -32002,
                "服务器未初始化"
            )

        tool_name = params.get("name")
        if not tool_name:
            return self._error_response(
                request_id,
                -32602,
                "无效参数：'name' 是必需的"
            )

        # TODO: 路由到工具实现（E3+ 将实现工具）
        return self._error_response(
            request_id,
            -32601,
            f"工具未找到：{tool_name}"
        )

    def _error_response(self, request_id: Optional[int], code: int, message: str) -> Dict[str, Any]:
        """创建 JSON-RPC 错误响应。"""
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
        主服务器循环：从 stdin 读取，写入 stdout。

        从 stdin 逐行读取 JSON-RPC 请求。
        向 stdout 逐行写入 JSON-RPC 响应。
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
                    # 解析 JSON-RPC 请求
                    request = json.loads(line)
                    logger.debug(f"Received request: {request}")

                    # 处理请求
                    response = self.handle_request(request)

                    # 将响应写入 stdout（如果不为 None）
                    if response is not None:
                        response_json = json.dumps(response)
                        print(response_json, flush=True)
                        logger.debug(f"Sent response: {response}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    error_response = self._error_response(
                        None,
                        -32700,
                        f"解析错误：{str(e)}"
                    )
                    print(json.dumps(error_response), flush=True)

                except Exception as e:
                    logger.error(f"Internal error: {e}", exc_info=True)
                    error_response = self._error_response(
                        None,
                        -32603,
                        "内部错误"
                    )
                    print(json.dumps(error_response), flush=True)

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            logger.info("MCP Server shutting down")


def main():
    """入口点。"""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
