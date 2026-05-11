"""
MCP 服务器的协议处理器。

处理 JSON-RPC 2.0 协议解析和工具路由。
"""

from typing import Any, Dict, Callable, List
import logging


logger = logging.getLogger(__name__)


class JSONRPCError(Exception):
    """带有错误码和消息的 JSON-RPC 错误。"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ProtocolHandler:
    """
    处理 MCP 协议操作。

    管理 JSON-RPC 2.0 请求的工具注册和路由。
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
        向协议处理器注册工具。

        Args:
            name: 工具名称（唯一标识符）
            description: 人类可读的描述
            input_schema: 工具输入的 JSON Schema
            handler: 实现工具的可调用对象
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
        处理 initialize 请求。

        Args:
            params: 初始化参数（clientInfo 等）

        Returns:
            包含 serverInfo 和 capabilities 的初始化结果
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
        处理 tools/list 请求。

        Returns:
            已注册工具的 schema 列表
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
        处理 tools/call 请求。

        路由到适当的工具处理器并执行。

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果

        Raises:
            JSONRPCError: 带有适当的错误码
        """
        # 检查工具是否存在
        if name not in self.tools:
            raise JSONRPCError(-32601, f"工具未找到：{name}")

        tool_def = self.tools[name]
        handler = tool_def["handler"]

        try:
            # 调用工具处理器
            result = handler(**arguments)
            return result

        except TypeError as e:
            # 无效参数（错误的参数类型/名称）
            logger.warning(f"Invalid parameters for tool {name}: {e}")
            raise JSONRPCError(-32602, "无效参数")

        except Exception as e:
            # 内部错误 - 不泄露详细信息
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            raise JSONRPCError(-32603, "内部错误")
