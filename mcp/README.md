# Boom MCP Server

MCP (Model Context Protocol) server for Boom - allows AI agents like Claude to interact with your reading library.

## Coming Soon

This will wrap the Boom core library so AI agents can:
- Search your book library
- Query books with RAG
- Get reading recommendations
- Access your reading history

## Structure (Future)

```
mcp/
├── src/
│   ├── server.py      # MCP server
│   └── tools.py       # Tool definitions
└── pyproject.toml     # Dependencies
```

The MCP server will import from `../backend/boom/` to access the core RAG functionality.
