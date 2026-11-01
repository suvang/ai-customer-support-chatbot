import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from store_support.memory import Turn
from store_support.reply import LocalReplyWriter
from store_support.tools import TOOL_SCHEMAS, StoreTools, describe_call
from store_support.types import Citation, Customer, Facts, Order, Passage, Reply, Step

SYSTEM = (
    "You are the Store support assistant. Call a tool before you state a fact from the Store or a Store policy. "
    "Use find_customers for a Customer's name, email, or city. "
    "Use find_products for a Product's name, sku, category, or description. "
    "Use find_order_lines for a price or a quantity. "
    "Use get_order when the Customer mentions an Order number, including phrases like 'order number 1048'. "
    "Use find_orders for what someone ordered, who ordered a Product, or Orders with a status. "
    "Use search_support_documents for refunds, shipping delays, and what an Order status means. "
    "Each tool reads one table. Call more than one when the question needs more than one table. "
    "Answer only from tool results. If a tool returns nothing, say so and offer a person on the Store support team. "
    "Do not invent a Customer, a Product, a price, a date, a use, or a policy. "
    "State unit_price_cents as a dollar amount. "
    "Mention every record a tool returned. "
    "When a Support document supports a sentence, name that document. "
    "Do not mention payment methods. "
    "Earlier turns of this conversation come before the newest message. Use them to resolve words like "
    "'he', 'she', 'they', 'it', or 'that order' to the Customer or Order they refer to, then call a tool for "
    "fresh facts. Do not ask who the Customer means when the conversation already names them. "
    "For 'last' or 'latest', pick the Order with the most recent placed_on."
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: tuple[ToolCall, ...]


class ToolSession:
    def start(
        self,
        system: str,
        user: str,
        tools: list[dict[str, object]],
        history: tuple[Turn, ...] = (),
    ) -> ModelTurn:
        raise NotImplementedError

    def resume(self, results: list[tuple[ToolCall, str]]) -> ModelTurn:
        raise NotImplementedError


class LangChainToolSession(ToolSession):
    def __init__(self, model: object) -> None:
        self._model = model
        self._bound: object | None = None
        self._messages: list[object] = []

    def start(
        self,
        system: str,
        user: str,
        tools: list[dict[str, object]],
        history: tuple[Turn, ...] = (),
    ) -> ModelTurn:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        bind_tools = getattr(self._model, "bind_tools")
        self._bound = bind_tools(tools)
        self._messages = [SystemMessage(content=system)]
        # Only the words of earlier turns are replayed, not their tool payloads: the model calls
        # tools again for fresh facts, and the prompt stays small.
        for turn in history:
            self._messages.append(HumanMessage(content=turn.user))
            self._messages.append(AIMessage(content=turn.assistant))
        self._messages.append(HumanMessage(content=user))
        return self._invoke()

    def resume(self, results: list[tuple[ToolCall, str]]) -> ModelTurn:
        from langchain_core.messages import ToolMessage

        for call, payload in results:
            self._messages.append(ToolMessage(content=payload, tool_call_id=call.id))
        return self._invoke()

    def _invoke(self) -> ModelTurn:
        if self._bound is None:
            raise RuntimeError("The tool session has not started.")
        invoke = getattr(self._bound, "invoke")
        result = invoke(self._messages)
        self._messages.append(result)
        calls: list[ToolCall] = []
        for call in getattr(result, "tool_calls", None) or []:
            arguments = call.get("args", {})
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                ToolCall(
                    id=str(call.get("id", "")),
                    name=str(call.get("name", "")),
                    arguments={str(key): value for key, value in arguments.items()},
                )
            )
        return ModelTurn(text=_message_text(result.content), tool_calls=tuple(calls))


class ToolAgent:
    def __init__(self, session_factory: Callable[[], ToolSession], tools: StoreTools) -> None:
        self._session_factory = session_factory
        self._tools = tools

    def ask(
        self,
        message: str,
        today: date,
        history: tuple[Turn, ...] = (),
        focus_note: str = "",
    ) -> Reply:
        session = self._session_factory()
        prompt = f"{SYSTEM}\nToday is {today.isoformat()}."
        if focus_note:
            prompt += f"\nThe conversation so far is about: {focus_note}."
        turn = session.start(prompt, message, TOOL_SCHEMAS, history)
        customers: list[Customer] = []
        orders: list[Order] = []
        passages: list[Passage] = []
        focus: Order | None = None
        grounded = False
        seen: set[str] = set()
        steps: list[Step] = []
        for _ in range(4):
            if not turn.tool_calls:
                break
            results: list[tuple[ToolCall, str]] = []
            for call in turn.tool_calls:
                outcome = self._tools.execute(call.name, call.arguments)
                steps.append(describe_call(call.name, call.arguments, outcome))
                grounded = grounded or outcome.grounded
                for order in outcome.orders:
                    if order.number in seen:
                        continue
                    orders.append(order)
                    seen.add(order.number)
                if outcome.focus is not None:
                    focus = outcome.focus
                passages.extend(outcome.passages)
                for customer in outcome.customers:
                    if customer not in customers:
                        customers.append(customer)
                results.append((call, json.dumps(outcome.payload)))
            turn = session.resume(results)
        text = turn.text.strip()
        order_tuple = tuple(orders)
        passage_tuple = tuple(_unique_passages(passages))
        if not text:
            text = LocalReplyWriter().write(
                message,
                Facts(orders=order_tuple, missing_order=None, focus=focus),
                passage_tuple,
                today,
            )
        return Reply(
            text=text,
            citations=tuple(
                Citation(document=passage.document, excerpt=passage.text) for passage in passage_tuple
            ),
            order=focus if focus is not None else (order_tuple[0] if len(order_tuple) == 1 else None),
            orders=order_tuple,
            escalation=bool(steps) and not grounded,
            steps=tuple(steps),
            customers=tuple(customers),
        )


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _unique_passages(passages: list[Passage]) -> list[Passage]:
    unique: list[Passage] = []
    seen: set[tuple[str, str]] = set()
    for passage in passages:
        key = (passage.document, passage.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(passage)
    return unique
