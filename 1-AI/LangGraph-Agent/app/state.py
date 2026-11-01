import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    intent: str
    approval_decision: str | None
    ticket_id: str | None
    trace: Annotated[list[str], operator.add]