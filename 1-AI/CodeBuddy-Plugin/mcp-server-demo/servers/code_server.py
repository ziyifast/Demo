# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]>=2.0.0"]
# ///
from mcp.server import MCPServer
import json

mcp = MCPServer("incident-code")

MOCK_CODE = {
    "OrderService.calculateTotal": [
        {
            "file": "src/main/java/com/example/OrderService.java",
            "line": 87,
            "snippet": "BigDecimal total = order.getItems().stream()\n    .map(item -> item.getPrice().multiply(item.getQuantity()))\n    .reduce(BigDecimal.ZERO, BigDecimal::add);\n// BUG: discount not applied before reduce"
        }
    ]
}

@mcp.tool()
def search_code(symbol: str) -> str:
    """Search the codebase by symbol name. Use this AFTER extracting signals from logs."""
    return json.dumps(MOCK_CODE.get(symbol, []), indent=2)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
