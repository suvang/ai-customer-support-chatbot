import re
from collections.abc import Callable
from dataclasses import replace
from datetime import date

from store_support.agent import ToolAgent
from store_support.library import SupportLibrary
from store_support.memory import Conversation, ConversationStore
from store_support.orders import OrderBook
from store_support.reply import GREETING, LocalReplyWriter, OpenAIReplyWriter
from store_support.types import Citation, Facts, Order, Reply, Step

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|hiya|yo|good (morning|afternoon|evening)|help|thanks|thank you|"
    r"what can you do|who are you)\b[\s!?.]*$",
    re.IGNORECASE,
)
# Words that point back at something said earlier in the conversation.
_REFERS_BACK = re.compile(
    r"\b(he|she|they|him|her|his|hers|them|their|theirs|it|its|that|this|those|these|same)\b"
    r"|^\s*(and|what about|how about)\b",
    re.IGNORECASE,
)
_LATEST = re.compile(r"\b(last|latest|most recent|newest|recent)\b", re.IGNORECASE)


class SupportAssistant:
    def __init__(
        self,
        library: SupportLibrary,
        orders: OrderBook,
        writer: LocalReplyWriter | OpenAIReplyWriter,
        agent: ToolAgent | None = None,
        clock: Callable[[], date] = date.today,
        model_name: str | None = None,
        memory: ConversationStore | None = None,
    ) -> None:
        self._library = library
        self._orders = orders
        self._writer = writer
        self._agent = agent
        self._clock = clock
        self._model_name = model_name
        self._memory = memory if memory is not None else ConversationStore()

    def ask(
        self,
        message: str,
        order_number: str | None = None,
        today: date | None = None,
        conversation_id: str | None = None,
    ) -> Reply:
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("Ask a question about an Order or a Store policy.")
        day = today or self._clock()
        number = (order_number or "").strip().lstrip("#")
        conversation = self._memory.open(conversation_id)
        if _GREETING.match(cleaned) and not number:
            reply = Reply(text=GREETING, citations=(), order=None, orders=(), escalation=False)
        elif self._agent is not None:
            hinted = f"{cleaned}\nOrder number: {number}" if number else cleaned
            reply = self._agent.ask(
                hinted,
                day,
                history=conversation.history(),
                focus_note=conversation.focus_note(),
            )
        else:
            reply = self._ask_locally(cleaned, number, day, conversation)
        self._memory.record(conversation, cleaned, reply)
        return replace(reply, conversation_id=conversation.id)

    def forget(self, conversation_id: str) -> None:
        self._memory.forget(conversation_id)

    def remembered(self, conversation_id: str) -> str:
        """The records this conversation is currently about, in a few words."""
        conversation = self._memory.peek(conversation_id)
        return conversation.focus_note() if conversation is not None else ""

    def _ask_locally(self, message: str, number: str, day: date, conversation: Conversation) -> Reply:
        facts = self._orders.lookup(message, number)
        steps: list[Step] = []
        if not facts.orders and not facts.missing_order and _REFERS_BACK.search(message):
            remembered = self._remembered_facts(message, conversation)
            if remembered is not None:
                facts = remembered
                steps.append(Step(label="Used conversation memory", detail=conversation.focus_note()))
        facts = _narrow_to_latest(message, facts)
        passages = self._library.search(message)
        text = self._writer.write(message, facts, passages, day)
        steps.append(_order_step(facts.orders, facts.missing_order))
        documents = sorted({passage.document for passage in passages})
        steps.append(
            Step(
                label="Searched the Support library",
                detail=", ".join(documents) if documents else "nothing relevant",
            )
        )
        return Reply(
            text=text,
            citations=tuple(Citation(document=passage.document, excerpt=passage.text) for passage in passages),
            order=facts.focus,
            orders=facts.orders,
            escalation=not passages and not facts.orders,
            steps=tuple(steps),
        )

    def _remembered_facts(self, message: str, conversation: Conversation) -> Facts | None:
        """Resolve 'he', 'it', 'that order' to the records the conversation was last about."""
        if len(conversation.order_numbers) == 1 and not _asks_about_a_person(message):
            return self._orders.lookup(message, conversation.order_numbers[0])
        if conversation.customers:
            facts = self._orders.lookup(f"{message} {' '.join(conversation.customers)}", "")
            if facts.orders:
                return facts
        if conversation.order_numbers:
            return self._orders.lookup(f"{message} {' '.join(conversation.order_numbers)}", "")
        return None

    def orders(self) -> tuple[Order, ...]:
        return self._orders.list()

    def examples(self) -> tuple[dict[str, str | None], ...]:
        return self._orders.examples()

    def documents(self) -> tuple[dict[str, str], ...]:
        return self._library.documents()

    def status(self) -> dict[str, str | None]:
        if self._agent is not None:
            return {"mode": "ai", "model": self._model_name}
        return {"mode": "local", "model": None}


def _asks_about_a_person(message: str) -> bool:
    return re.search(r"\b(he|she|they|him|her|his|them|their)\b", message, re.IGNORECASE) is not None


def _narrow_to_latest(message: str, facts: Facts) -> Facts:
    """'What did he order last?' means the most recent of the matched Orders."""
    if facts.focus is not None or len(facts.orders) < 2 or not _LATEST.search(message):
        return facts
    latest = max(facts.orders, key=lambda order: order.placed_on)
    return Facts(orders=(latest,), missing_order=None, focus=latest)


def _order_step(orders: tuple[Order, ...], missing: str | None) -> Step:
    if missing:
        return Step(label="Searched Order records", detail=f"Order {missing}: not found")
    if not orders:
        return Step(label="Searched Order records", detail="no matching Order")
    numbers = ", ".join(order.number for order in orders)
    noun = "Order" if len(orders) == 1 else "Orders"
    return Step(label="Searched Order records", detail=f"{noun} {numbers}")
