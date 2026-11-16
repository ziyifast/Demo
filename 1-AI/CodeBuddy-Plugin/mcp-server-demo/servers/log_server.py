# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]>=2.0.0"]
# ///
from mcp.server import MCPServer
import json

mcp = MCPServer("incident-logs")

MOCK_LOGS = {
    "trace-abc123": [
        {"level": "ERROR", "message": "NullPointerException at OrderService.calculateTotal(OrderService.java:87)"},
        {"level": "WARN", "message": "Retry attempt 1 for external payment gateway"},
        {"level": "ERROR", "message": "Failed to calculate order total for orderId=ORD-9981"},
    ]
}

@mcp.tool()
def query_logs(trace_id: str) -> str:
    """Query application logs by trace_id. Use this FIRST in any diagnosis."""
    return json.dumps(MOCK_LOGS.get(trace_id, []), indent=2)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
