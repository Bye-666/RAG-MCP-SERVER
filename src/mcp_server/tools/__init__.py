"""MCP Server tools."""

# Import tools with error handling to avoid dependency issues during testing
__all__ = []

try:
    from .query_knowledge_hub import query_knowledge_hub, get_tool_schema as query_schema
    __all__.extend(['query_knowledge_hub', 'query_schema'])
except ImportError:
    pass

try:
    from .list_collections import list_collections, get_tool_schema as list_collections_schema
    __all__.extend(['list_collections', 'list_collections_schema'])
except ImportError:
    pass
