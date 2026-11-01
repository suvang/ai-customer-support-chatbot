import json
import re
from datetime import date

from store_support.types import Facts, Order, OrderLine, Passage

_WINDOW = re.compile(r"(\d+)\s+days", re.IGNORECASE)
_REFUND = re.compile(r"refund", re.IGNORECASE)
_POLICY = re.compile(r"refund|policy|shipping delay|eligible", re.IGNORECASE)

ESCALATION = (
    "I don't have a Support document for that, and I don't have an Order record to read. "
    "A person on the Store support team can help."
)

GREETING = (
    "Hi! I'm the Store support assistant. I can look up an Order (what's in it, where it is, "
    "whether it's late), check Refund eligibility, and explain Store policy. "
    "Try asking \"Is order 1042 delayed?\" or \"Can I get a refund for order 1048?\""
)

GROUNDED_INSTRUCTION = (
    "You are the Store support assistant. Write a short reply to the Customer. "
    "Use only the Order records and the Support documents in the user message. "
    "If support_documents is not empty, answer from those documents and name each document you use. "
    "If order_records is not empty, answer from those records, including the Customer, "
    "each Product, the quantity, and the price. "
    "Mention every Order in order_records. Do not leave one out. "
    "Do not invent a Customer, a Product, or an Order line that is not in the records. "
    "A null order_record means you must not invent an Order status or any Order date. "
    "It does not mean the question is unanswerable. "
    "Say a person on the Store support team can help only when support_documents is empty "
    "and order_records is empty. "
    "Do not mention payment methods."
)


class LocalReplyWriter:
    def write(
        self,
        message: str,
        facts: Facts,
        passages: tuple[Passage, ...],
        today: date,
    ) -> str:
        parts: list[str] = []
        if facts.missing_order:
            parts.append(
                f"Order {facts.missing_order} is not in the Store's records. I can't state an Order status for it."
            )
        for order in facts.orders:
            parts.append(_order_facts(order))
        if passages:
            verdicts = [
                verdict
                for verdict in (
                    _refund_eligibility(message, facts.focus, passages, today),
                    _shipping_delay(facts.focus, passages, today),
                )
                if verdict
            ]
            parts = verdicts + parts
            parts.append(_policy_reference(passages) if verdicts else _policy_facts(passages))
        elif not facts.orders and not facts.missing_order:
            parts.append(ESCALATION)
        elif _POLICY.search(message):
            parts.append(
                "I don't have a Support document for the policy side of this. I won't guess Store policy."
            )
        return "\n\n".join(parts)


class OpenAIReplyWriter:
    def __init__(self, chat: object) -> None:
        self._chat = chat

    def write(
        self,
        message: str,
        facts: Facts,
        passages: tuple[Passage, ...],
        today: date,
    ) -> str:
        payload = {
            "today": today.isoformat(),
            "customer_message": message,
            "missing_order_number": facts.missing_order,
            "order_record": _order_payload(facts.focus),
            "order_records": [_order_payload(order) for order in facts.orders],
            "support_documents": [
                {"document": passage.document, "text": passage.text} for passage in passages
            ],
        }
        messages = [
            {"role": "system", "content": GROUNDED_INSTRUCTION},
            {"role": "user", "content": json.dumps(payload)},
        ]
        complete = getattr(self._chat, "complete")
        return str(complete(messages))


class LangChainChat:
    def __init__(self, model: object) -> None:
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        converted: list[HumanMessage | SystemMessage] = []
        for message in messages:
            if message["role"] == "system":
                converted.append(SystemMessage(content=message["content"]))
            else:
                converted.append(HumanMessage(content=message["content"]))
        invoke = getattr(self._model, "invoke")
        result = invoke(converted)
        return str(result.content)


def _order_facts(order: Order) -> str:
    sentences = [
        f"Order {order.number} is {order.status}.",
        f"It was placed on {order.placed_on.isoformat()}.",
    ]
    if order.customer is not None:
        sentences.append(
            f"{order.customer.name} in {order.customer.city} placed it ({order.customer.email})."
        )
    if order.promised_on is not None:
        sentences.append(f"The Store promised delivery on {order.promised_on.isoformat()}.")
    if order.delivered_on is not None:
        sentences.append(f"It was delivered on {order.delivered_on.isoformat()}.")
    else:
        sentences.append("It has not been delivered.")
    text = " ".join(sentences)
    if order.lines:
        text += " It contains:\n" + "\n".join(f"- {_line_text(line)}" for line in order.lines)
    return text


def _line_text(line: OrderLine) -> str:
    price = f"${line.unit_price_cents // 100}.{line.unit_price_cents % 100:02d}"
    return (
        f"{line.quantity} × {line.product.name} ({line.product.description}) at {price} each"
    )


def _policy_facts(passages: tuple[Passage, ...]) -> str:
    lines: list[str] = []
    for passage in passages:
        title = passage.document
        body: list[str] = []
        for line in passage.text.splitlines():
            if line.startswith("#"):
                title = line.lstrip("#").strip()
            elif line.strip():
                body.append(line.strip())
        lines.append(f"From **{title}** ({passage.document}): {' '.join(body)}")
    return "\n\n".join(lines)


def _policy_reference(passages: tuple[Passage, ...]) -> str:
    names = list(dict.fromkeys(f"**{_passage_title(passage)}** ({passage.document})" for passage in passages))
    return "Based on " + " and ".join(names) + "."


def _passage_title(passage: Passage) -> str:
    for line in passage.text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return passage.document


def _shipping_delay(order: Order | None, passages: tuple[Passage, ...], today: date) -> str | None:
    if order is None:
        return None
    blob = "\n".join(passage.text for passage in passages).lower()
    if "shipping delay" not in blob or "promised" not in blob:
        return None
    if order.status == "cancelled" or order.promised_on is None:
        return None
    if order.delivered_on is not None or order.status == "delivered":
        return f"Order {order.number} has arrived, so it is not a Shipping delay."
    gap = (today - order.promised_on).days
    if gap > 0:
        noun = "day" if gap == 1 else "days"
        return (
            f"**Order {order.number} is delayed.** The Store promised delivery on "
            f"{order.promised_on.isoformat()}, {gap} {noun} ago, and it has not arrived yet."
        )
    remaining = -gap
    when = "today" if remaining == 0 else f"in {remaining} day{'s' if remaining != 1 else ''}"
    return (
        f"**Order {order.number} is not delayed.** The Store promised delivery on "
        f"{order.promised_on.isoformat()} ({when})."
    )


def _refund_eligibility(
    message: str,
    order: Order | None,
    passages: tuple[Passage, ...],
    today: date,
) -> str | None:
    if order is None or _REFUND.search(message) is None:
        return None
    blob = "\n".join(passage.text for passage in passages)
    if _REFUND.search(blob) is None:
        return None
    window = _WINDOW.search(blob)
    if window is None:
        return None
    days = int(window.group(1))
    if order.status == "cancelled" and "cancelled" in blob.lower():
        return (
            f"Order {order.number} is cancelled, so under the Support documents it cannot be refunded."
        )
    if order.delivered_on is None and "has not been delivered" in blob.lower():
        return (
            f"Order {order.number} has not been delivered, so under the Support documents a Refund is not available yet."
        )
    if order.delivered_on is None:
        return None
    age = (today - order.delivered_on).days
    if age <= days:
        return (
            f"Order {order.number} was delivered on {order.delivered_on.isoformat()}, "
            f"which is {age} days ago. The Support documents allow a Refund for {days} days after delivery, "
            "so a Refund is available."
        )
    return (
        f"Order {order.number} was delivered on {order.delivered_on.isoformat()}, "
        f"which is {age} days ago. The Support documents allow a Refund for {days} days after delivery, "
        "so a Refund is not available."
    )


def _order_payload(order: Order | None) -> dict[str, object] | None:
    if order is None:
        return None
    customer: dict[str, str] | None = None
    if order.customer is not None:
        customer = {
            "name": order.customer.name,
            "email": order.customer.email,
            "city": order.customer.city,
        }
    return {
        "number": order.number,
        "status": order.status,
        "placed_on": order.placed_on.isoformat(),
        "promised_on": order.promised_on.isoformat() if order.promised_on else None,
        "delivered_on": order.delivered_on.isoformat() if order.delivered_on else None,
        "customer": customer,
        "lines": [
            {
                "sku": line.product.sku,
                "name": line.product.name,
                "description": line.product.description,
                "quantity": line.quantity,
                "unit_price_cents": line.unit_price_cents,
            }
            for line in order.lines
        ],
    }
