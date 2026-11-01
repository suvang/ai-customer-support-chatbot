"""Conversation memory: what was said recently and which records it was about.

Kept on the server, keyed by a conversation id the client sends back with each message.
Bounded on every axis so it stays cheap: a few recent turns per conversation, a capped
number of conversations (least recently used is dropped first), and an idle timeout.
"""

import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from store_support.types import Reply

MAX_TURNS = 6
MAX_REPLY_CHARS = 1200


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass
class Conversation:
    id: str
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    order_numbers: tuple[str, ...] = ()
    customers: tuple[str, ...] = ()
    touched: float = 0.0

    def history(self) -> tuple[Turn, ...]:
        return tuple(self.turns)

    def focus_note(self) -> str:
        """One line naming the records the last replies were about, for resolving 'he', 'it', 'that order'."""
        parts: list[str] = []
        if self.customers:
            parts.append("Customer " + ", ".join(self.customers))
        if self.order_numbers:
            parts.append("Order " + ", ".join(self.order_numbers))
        return "; ".join(parts)


class ConversationStore:
    def __init__(
        self,
        max_conversations: int = 500,
        idle_seconds: float = 2 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._max = max_conversations
        self._idle = idle_seconds
        self._clock = clock
        self._lock = threading.Lock()

    def open(self, conversation_id: str | None) -> Conversation:
        """Return the conversation for this id, or start a new one if it is unknown or expired."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            if conversation_id and conversation_id in self._items:
                conversation = self._items[conversation_id]
                self._items.move_to_end(conversation_id)
            else:
                conversation = Conversation(id=uuid.uuid4().hex)
                self._items[conversation.id] = conversation
                while len(self._items) > self._max:
                    self._items.popitem(last=False)
            conversation.touched = now
            return conversation

    def record(self, conversation: Conversation, message: str, reply: Reply) -> None:
        text = reply.text if len(reply.text) <= MAX_REPLY_CHARS else reply.text[:MAX_REPLY_CHARS] + "…"
        with self._lock:
            conversation.turns.append(Turn(user=message, assistant=text))
            numbers = tuple(order.number for order in reply.orders)
            names = list(customer.name for customer in reply.customers)
            for order in reply.orders:
                if order.customer is not None and order.customer.name not in names:
                    names.append(order.customer.name)
            # Only move the focus when this reply was about records; a policy answer keeps the old focus.
            if numbers or names:
                conversation.order_numbers = numbers
                conversation.customers = tuple(names)
            conversation.touched = self._clock()

    def peek(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._items.get(conversation_id)

    def forget(self, conversation_id: str) -> None:
        with self._lock:
            self._items.pop(conversation_id, None)

    def __len__(self) -> int:
        return len(self._items)

    def _expire(self, now: float) -> None:
        stale = [key for key, item in self._items.items() if now - item.touched > self._idle]
        for key in stale:
            del self._items[key]
