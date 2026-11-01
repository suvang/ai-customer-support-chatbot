from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Customer:
    name: str
    email: str
    city: str


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    description: str


@dataclass(frozen=True)
class OrderLine:
    product: Product
    quantity: int
    unit_price_cents: int


@dataclass(frozen=True)
class Order:
    number: str
    status: str
    placed_on: date
    promised_on: date | None
    delivered_on: date | None
    customer: Customer | None = None
    lines: tuple[OrderLine, ...] = ()


@dataclass(frozen=True)
class Facts:
    orders: tuple[Order, ...]
    missing_order: str | None
    focus: Order | None


@dataclass(frozen=True)
class Passage:
    document: str
    text: str


@dataclass(frozen=True)
class Citation:
    document: str
    excerpt: str


@dataclass(frozen=True)
class Step:
    """One lookup the assistant made to build a Reply, shown to the Customer as a trace."""

    label: str
    detail: str


@dataclass(frozen=True)
class Reply:
    text: str
    citations: tuple[Citation, ...]
    order: Order | None
    orders: tuple[Order, ...]
    escalation: bool
    steps: tuple[Step, ...] = ()
    customers: tuple[Customer, ...] = ()
    conversation_id: str | None = None
