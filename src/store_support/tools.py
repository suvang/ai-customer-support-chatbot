import json
from dataclasses import dataclass

from store_support.library import SupportLibrary
from store_support.orders import OrderBook
from store_support.reply import _order_payload
from store_support.types import Customer, Order, OrderLine, Passage, Product, Step

TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "name": "get_order",
        "description": "Load one Order by its number, including the Customer who placed it and each Order line.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "string",
                    "description": "The Order number, such as 1048.",
                }
            },
            "required": ["number"],
        },
    },
    {
        "name": "find_customers",
        "description": (
            "Read the customers table. "
            "Use this for who a Customer is, their email, or their city. "
            "Leave every filter empty to list every Customer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A Customer name, or part of one. Empty string if unused.",
                },
                "email": {
                    "type": "string",
                    "description": "An email address, or part of one. Empty string if unused.",
                },
                "city": {
                    "type": "string",
                    "description": "A city, or part of one. Empty string if unused.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_products",
        "description": (
            "Read the products table: sku, name, category, and description. "
            "Use this when the question is about a Product itself, not about who bought it. "
            "Price and quantity are on order lines, not on this table. "
            "Leave every filter empty to list every Product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A Product name, or part of one. Empty string if unused.",
                },
                "sku": {
                    "type": "string",
                    "description": "A Product sku, or part of one. Empty string if unused.",
                },
                "category": {
                    "type": "string",
                    "description": "A category such as Kitchen or Home. Empty string if unused.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_orders",
        "description": (
            "Read the orders table. "
            "Filter by Customer name, Product name, or Order status. "
            "Use this for what someone ordered, who ordered a Product, or which Orders have a status. "
            "Leave every filter empty to list every Order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "string",
                    "description": "A Customer name, or part of one. Empty string if unused.",
                },
                "product": {
                    "type": "string",
                    "description": "A Product name, or part of one. Empty string if unused.",
                },
                "status": {
                    "type": "string",
                    "description": "An Order status such as delivered, shipped, packed, placed, or cancelled. Empty string if unused.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_order_lines",
        "description": (
            "Read the order_lines table: quantity and unit price for a Product on an Order. "
            "Use this for a price, a quantity, or which Orders include a Product. "
            "Leave every filter empty to list every line."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": {
                    "type": "string",
                    "description": "An Order number. Empty string if unused.",
                },
                "product": {
                    "type": "string",
                    "description": "A Product name, or part of one. Empty string if unused.",
                },
                "sku": {
                    "type": "string",
                    "description": "A Product sku, or part of one. Empty string if unused.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_support_documents",
        "description": (
            "Search Store policy and FAQ Support documents. "
            "Use this for refunds, shipping delays, and what an Order status means."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The policy question, in the Customer's words.",
                }
            },
            "required": ["query"],
        },
    },
]


@dataclass(frozen=True)
class ToolOutcome:
    payload: dict[str, object]
    orders: tuple[Order, ...] = ()
    passages: tuple[Passage, ...] = ()
    focus: Order | None = None
    grounded: bool = False
    customers: tuple[Customer, ...] = ()


class StoreTools:
    """The only way the model reads the Store. Each method runs a fixed query."""

    def __init__(self, orders: OrderBook, library: SupportLibrary) -> None:
        self._orders = orders
        self._library = library

    def execute(self, name: str, arguments: object) -> ToolOutcome:
        args = _arguments(arguments)
        if name == "get_order":
            return self._get_order(str(args.get("number", "")))
        if name == "find_customers":
            return self._find_customers(
                str(args.get("name", "")),
                str(args.get("email", "")),
                str(args.get("city", "")),
            )
        if name == "find_products":
            return self._find_products(
                str(args.get("name", "")),
                str(args.get("sku", "")),
                str(args.get("category", "")),
            )
        if name == "find_orders":
            return self._find_orders(
                str(args.get("customer", "")),
                str(args.get("product", "")),
                str(args.get("status", "")),
            )
        if name == "find_order_lines":
            return self._find_lines(
                str(args.get("order_number", "")),
                str(args.get("product", "")),
                str(args.get("sku", "")),
            )
        if name == "search_support_documents":
            return self._search(str(args.get("query", "")))
        return ToolOutcome({"error": f"Unknown tool {name}."}, (), ())

    def _get_order(self, number: str) -> ToolOutcome:
        cleaned = number.strip().lstrip("#")
        order = self._orders.get(cleaned) if cleaned else None
        if order is None:
            return ToolOutcome({"found": False, "number": cleaned}, (), ())
        payload = _order_payload(order)
        return ToolOutcome(
            {"found": True, "order": payload if payload is not None else {}},
            (order,),
            (),
            focus=order,
            grounded=True,
        )

    def _find_customers(self, name: str, email: str, city: str) -> ToolOutcome:
        customers = self._orders.find_customers(name.strip(), email.strip(), city.strip())
        return ToolOutcome(
            {"customers": [_customer_payload(customer) for customer in customers]},
            grounded=bool(customers),
            customers=customers,
        )

    def _find_products(self, name: str, sku: str, category: str) -> ToolOutcome:
        products = self._orders.find_products(name.strip(), sku.strip(), category.strip())
        return ToolOutcome(
            {"products": [_product_payload(product) for product in products]},
            grounded=bool(products),
        )

    def _find_orders(self, customer: str, product: str, status: str) -> ToolOutcome:
        customer_name = customer.strip().lower()
        product_name = product.strip().lower()
        status_name = status.strip().lower()
        matched: list[Order] = []
        for order in self._orders.list():
            customer_ok = not customer_name or (
                order.customer is not None and customer_name in order.customer.name.lower()
            )
            product_ok = not product_name or any(
                product_name in line.product.name.lower() for line in order.lines
            )
            status_ok = not status_name or status_name == order.status.lower()
            if customer_ok and product_ok and status_ok:
                matched.append(order)
        payloads = [payload for order in matched if (payload := _order_payload(order)) is not None]
        return ToolOutcome({"orders": payloads}, tuple(matched), (), grounded=bool(matched))

    def _find_lines(self, order_number: str, product: str, sku: str) -> ToolOutcome:
        lines = self._orders.find_lines(order_number.strip().lstrip("#"), product.strip(), sku.strip())
        return ToolOutcome(
            {"lines": [_line_payload(number, line) for number, line in lines]},
            grounded=bool(lines),
        )

    def _search(self, query: str) -> ToolOutcome:
        passages = self._library.search(query.strip())
        return ToolOutcome(
            {
                "documents": [
                    {"document": passage.document, "text": passage.text} for passage in passages
                ]
            },
            (),
            passages,
            grounded=bool(passages),
        )


_STEP_LABELS = {
    "get_order": "Opened an Order record",
    "find_customers": "Searched Customers",
    "find_products": "Searched Products",
    "find_orders": "Searched Orders",
    "find_order_lines": "Searched Order lines",
    "search_support_documents": "Searched the Support library",
}


def describe_call(name: str, arguments: object, outcome: ToolOutcome) -> Step:
    """A plain-language trace of one tool call, so the Customer can see where an answer came from."""
    args = _arguments(arguments)
    filters = ", ".join(
        f"{key} “{value}”" for key, value in args.items() if isinstance(value, str) and value.strip()
    )
    label = _STEP_LABELS.get(name, f"Ran {name}")
    payload = outcome.payload
    if name == "get_order":
        found = "found" if payload.get("found") else "not found"
        detail = f"Order {args.get('number', '')}: {found}"
    elif name == "search_support_documents":
        names = sorted({passage.document for passage in outcome.passages})
        detail = f"for “{args.get('query', '')}”: " + (", ".join(names) if names else "nothing relevant")
    else:
        count = 0
        for value in payload.values():
            if isinstance(value, list):
                count = len(value)
        noun = "result" if count == 1 else "results"
        detail = f"{filters or 'no filter'}: {count} {noun}"
    return Step(label=label, detail=detail)


def _customer_payload(customer: Customer) -> dict[str, str]:
    return {"name": customer.name, "email": customer.email, "city": customer.city}


def _product_payload(product: Product) -> dict[str, str]:
    return {
        "sku": product.sku,
        "name": product.name,
        "category": product.category,
        "description": product.description,
    }


def _line_payload(order_number: str, line: OrderLine) -> dict[str, object]:
    return {
        "order_number": order_number,
        "sku": line.product.sku,
        "name": line.product.name,
        "category": line.product.category,
        "description": line.product.description,
        "quantity": line.quantity,
        "unit_price_cents": line.unit_price_cents,
    }


def _arguments(value: object) -> dict[str, object]:
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {str(key): item for key, item in parsed.items()}
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}
