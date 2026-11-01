import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from store_support.types import Customer, Facts, Order, OrderLine, Product

_WORD = re.compile(r"[a-z0-9]+")
_EXPLICIT_ORDER = re.compile(
    r"\border\s*(?:number|no\.?|#)?\s*(\d{3,6})\b",
    re.IGNORECASE,
)
_BARE_NUMBER = re.compile(r"\b(\d{3,6})\b")
_ALL_ORDERS = re.compile(r"\b(all|every|list)\b.*\borders?\b|\borders\b.*\b(all|every)\b")


class OrderBook:
    """Facts about Customers, Products, and Orders. The assistant reads these and does not invent them."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                city TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                number TEXT PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                status TEXT NOT NULL,
                placed_on TEXT NOT NULL,
                promised_on TEXT,
                delivered_on TEXT
            );
            CREATE TABLE IF NOT EXISTS order_lines (
                order_number TEXT NOT NULL REFERENCES orders(number),
                sku TEXT NOT NULL REFERENCES products(sku),
                quantity INTEGER NOT NULL,
                unit_price_cents INTEGER NOT NULL,
                PRIMARY KEY (order_number, sku)
            );
            """
        )
        self._conn.commit()

    def add(self, order: Order) -> None:
        customer_id = self._save_customer(order.customer)
        self._conn.execute(
            """
            INSERT INTO orders (number, customer_id, status, placed_on, promised_on, delivered_on)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(number) DO UPDATE SET
                customer_id = excluded.customer_id,
                status = excluded.status,
                placed_on = excluded.placed_on,
                promised_on = excluded.promised_on,
                delivered_on = excluded.delivered_on
            """,
            (
                order.number,
                customer_id,
                order.status,
                order.placed_on.isoformat(),
                _iso(order.promised_on),
                _iso(order.delivered_on),
            ),
        )
        self._conn.execute("DELETE FROM order_lines WHERE order_number = ?", (order.number,))
        for line in order.lines:
            self._save_product(line.product)
            self._conn.execute(
                """
                INSERT INTO order_lines (order_number, sku, quantity, unit_price_cents)
                VALUES (?, ?, ?, ?)
                """,
                (order.number, line.product.sku, line.quantity, line.unit_price_cents),
            )
        self._conn.commit()

    def get(self, number: str) -> Order | None:
        row = self._conn.execute(
            """
            SELECT orders.number, orders.status, orders.placed_on, orders.promised_on, orders.delivered_on,
                   customers.name AS customer_name, customers.email, customers.city
            FROM orders
            LEFT JOIN customers ON customers.id = orders.customer_id
            WHERE orders.number = ?
            """,
            (number,),
        ).fetchone()
        if row is None:
            return None
        return _order_from_row(row, self._lines(number))

    def list(self) -> tuple[Order, ...]:
        rows = self._conn.execute("SELECT number FROM orders ORDER BY number").fetchall()
        return tuple(order for row in rows if (order := self.get(str(row["number"]))) is not None)

    def find_customers(self, name: str = "", email: str = "", city: str = "") -> tuple[Customer, ...]:
        rows = self._conn.execute(
            """
            SELECT name, email, city FROM customers
            WHERE (? = '' OR instr(lower(name), lower(?)) > 0)
              AND (? = '' OR instr(lower(email), lower(?)) > 0)
              AND (? = '' OR instr(lower(city), lower(?)) > 0)
            ORDER BY name
            """,
            (name, name, email, email, city, city),
        ).fetchall()
        return tuple(
            Customer(name=str(row["name"]), email=str(row["email"]), city=str(row["city"]))
            for row in rows
        )

    def find_products(self, name: str = "", sku: str = "", category: str = "") -> tuple[Product, ...]:
        rows = self._conn.execute(
            """
            SELECT sku, name, category, description FROM products
            WHERE (? = '' OR instr(lower(name), lower(?)) > 0)
              AND (? = '' OR instr(lower(sku), lower(?)) > 0)
              AND (? = '' OR instr(lower(category), lower(?)) > 0)
            ORDER BY name
            """,
            (name, name, sku, sku, category, category),
        ).fetchall()
        return tuple(
            Product(
                sku=str(row["sku"]),
                name=str(row["name"]),
                category=str(row["category"]),
                description=str(row["description"]),
            )
            for row in rows
        )

    def find_lines(
        self,
        order_number: str = "",
        product: str = "",
        sku: str = "",
    ) -> tuple[tuple[str, OrderLine], ...]:
        rows = self._conn.execute(
            """
            SELECT order_lines.order_number, order_lines.quantity, order_lines.unit_price_cents,
                   products.sku, products.name, products.category, products.description
            FROM order_lines
            JOIN products ON products.sku = order_lines.sku
            WHERE (? = '' OR order_lines.order_number = ?)
              AND (? = '' OR instr(lower(products.name), lower(?)) > 0)
              AND (? = '' OR instr(lower(products.sku), lower(?)) > 0)
            ORDER BY order_lines.order_number, products.name
            """,
            (order_number, order_number, product, product, sku, sku),
        ).fetchall()
        return tuple(
            (
                str(row["order_number"]),
                OrderLine(
                    product=Product(
                        sku=str(row["sku"]),
                        name=str(row["name"]),
                        category=str(row["category"]),
                        description=str(row["description"]),
                    ),
                    quantity=int(row["quantity"]),
                    unit_price_cents=int(row["unit_price_cents"]),
                ),
            )
            for row in rows
        )

    def lookup(self, message: str, order_number: str | None) -> Facts:
        explicit = _explicit_order_number(message) or (order_number or "").strip()
        known_numbers = [number for number in _bare_numbers(message) if self.get(number) is not None]
        focus_number = explicit or (known_numbers[0] if known_numbers else "")
        focus = self.get(focus_number) if focus_number else None
        missing = focus_number if focus_number and focus is None else None
        matched: list[Order] = []
        seen: set[str] = set()
        if focus is not None:
            matched.append(focus)
            seen.add(focus.number)
        for number in known_numbers:
            if number in seen:
                continue
            order = self.get(number)
            if order is None:
                continue
            matched.append(order)
            seen.add(order.number)
        lowered = message.lower()
        list_all = _ALL_ORDERS.search(lowered) is not None and focus is None and not known_numbers
        for order in self.list():
            if order.number in seen:
                continue
            if list_all or _message_names(order, lowered):
                matched.append(order)
                seen.add(order.number)
        return Facts(orders=tuple(matched), missing_order=missing, focus=focus)

    def examples(self) -> tuple[dict[str, str | None], ...]:
        def item(group: str, label: str, message: str, order_number: str | None = None) -> dict[str, str | None]:
            return {"group": group, "label": label, "message": message, "order_number": order_number}

        return (
            item("Orders", "What's in order 1042?", "What is in order 1042?", "1042"),
            item("Orders", "Where is order 1072?", "Where is order 1072?", "1072"),
            item("Orders", "Show me all orders", "Show me all orders"),
            item("Shipping", "Is order 1042 delayed?", "Is order 1042 delayed?", "1042"),
            item("Shipping", "What does “packed” mean?", "What does the packed Order status mean?"),
            item("Refunds", "Refund for order 1048?", "Can I get a refund for order 1048?", "1048"),
            item("Refunds", "Refund for order 1001?", "Can I get a refund for order 1001?", "1001"),
            item("Refunds", "What is the refund policy?", "What is the refund policy?"),
            item("Customers & products", "Who ordered the kettle?", "Which Customer ordered the Cedar Pour-Over Kettle?"),
            item("Customers & products", "What did Mira order?", "What did Mira Ellison order?"),
            item("Customers & products", "Who is in Brooklyn?", "Which Customer lives in Brooklyn?"),
            item("Customers & products", "Price of the candle", "How much is the Night Market Candle?"),
        )

    def _lines(self, number: str) -> tuple[OrderLine, ...]:
        rows = self._conn.execute(
            """
            SELECT products.sku, products.name, products.category, products.description,
                   order_lines.quantity, order_lines.unit_price_cents
            FROM order_lines
            JOIN products ON products.sku = order_lines.sku
            WHERE order_lines.order_number = ?
            ORDER BY products.name
            """,
            (number,),
        ).fetchall()
        return tuple(
            OrderLine(
                product=Product(
                    sku=str(row["sku"]),
                    name=str(row["name"]),
                    category=str(row["category"]),
                    description=str(row["description"]),
                ),
                quantity=int(row["quantity"]),
                unit_price_cents=int(row["unit_price_cents"]),
            )
            for row in rows
        )

    def _save_customer(self, customer: Customer | None) -> int | None:
        if customer is None:
            return None
        self._conn.execute(
            """
            INSERT INTO customers (name, email, city) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET email = excluded.email, city = excluded.city
            """,
            (customer.name, customer.email, customer.city),
        )
        row = self._conn.execute(
            "SELECT id FROM customers WHERE name = ?",
            (customer.name,),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"])

    def _save_product(self, product: Product) -> None:
        self._conn.execute(
            """
            INSERT INTO products (sku, name, category, description) VALUES (?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                name = excluded.name,
                category = excluded.category,
                description = excluded.description
            """,
            (product.sku, product.name, product.category, product.description),
        )


def seed_demo_orders(book: OrderBook, today: date) -> None:
    mira = Customer("Mira Ellison", "mira.ellison@example.com", "Portland")
    jonah = Customer("Jonah Park", "jonah.park@example.com", "Austin")
    adele = Customer("Adele Voss", "adele.voss@example.com", "Chicago")
    samir = Customer("Samir Shah", "samir.shah@example.com", "Brooklyn")
    kettle = Product("KETTLE", "Cedar Pour-Over Kettle", "Kitchen", "A stainless kettle with a gooseneck spout.")
    carafe = Product("CARAFE", "Glass Carafe", "Kitchen", "A heatproof carafe marked in milliliters.")
    throw = Product("THROW", "Oat Linen Throw", "Home", "A washed linen throw in oat.")
    socks = Product("SOCKS", "Harbor Wool Socks", "Apparel", "Crew socks in harbor blue wool.")
    candle = Product("CANDLE", "Night Market Candle", "Home", "A soy candle with smoked fig and orange peel.")
    notes = Product("NOTES", "Field Notes Notebook", "Desk", "A pocket notebook with a kraft cover.")
    mug = Product("MUG", "Stoneware Mug", "Kitchen", "A speckled stoneware mug, 12 ounces.")
    tumbler = Product("TUMBLER", "Travel Tumbler", "Kitchen", "A leak-resistant tumbler for the commute.")

    book.add(_order("1001", "cancelled", mira, today, placed=10, promised=5, delivered=None, lines=((candle, 2, 3200),)))
    book.add(_order("1030", "delivered", jonah, today, placed=80, promised=75, delivered=70, lines=((socks, 3, 2400), (mug, 1, 2600))))
    book.add(_order("1042", "shipped", adele, today, placed=16, promised=6, delivered=None, lines=((kettle, 1, 6400), (carafe, 1, 2800))))
    book.add(_order("1048", "delivered", samir, today, placed=25, promised=21, delivered=20, lines=((throw, 1, 8800), (notes, 2, 1800))))
    book.add(_order("1055", "delivered", mira, today, placed=18, promised=14, delivered=12, lines=((kettle, 1, 6400), (tumbler, 2, 3400))))
    book.add(_order("1060", "packed", jonah, today, placed=2, promised=-3, delivered=None, lines=((throw, 2, 8800),)))
    book.add(_order("1072", "placed", adele, today, placed=1, promised=-6, delivered=None, lines=((candle, 4, 3200), (socks, 1, 2400))))


def _order(
    number: str,
    status: str,
    customer: Customer,
    today: date,
    placed: int,
    promised: int,
    delivered: int | None,
    lines: tuple[tuple[Product, int, int], ...],
) -> Order:
    return Order(
        number=number,
        status=status,
        placed_on=today - timedelta(days=placed),
        promised_on=today - timedelta(days=promised),
        delivered_on=None if delivered is None else today - timedelta(days=delivered),
        customer=customer,
        lines=tuple(
            OrderLine(product=product, quantity=quantity, unit_price_cents=price)
            for product, quantity, price in lines
        ),
    )


def _explicit_order_number(message: str) -> str:
    match = _EXPLICIT_ORDER.search(message)
    if match is None:
        return ""
    return match.group(1)


def _bare_numbers(message: str) -> list[str]:
    return list(dict.fromkeys(_BARE_NUMBER.findall(message)))


def _message_names(order: Order, message: str) -> bool:
    if order.customer is not None:
        customer = order.customer
        for name in (customer.name, *customer.name.split(), customer.city):
            if _mentions(message, name):
                return True
    for line in order.lines:
        if _mentions(message, line.product.name):
            return True
        for token in _WORD.findall(line.product.name.lower()):
            if len(token) >= 5 and _mentions(message, token):
                return True
    return False


def _mentions(message: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase.lower())}\b", message) is not None


def _order_from_row(row: sqlite3.Row, lines: tuple[OrderLine, ...]) -> Order:
    customer: Customer | None = None
    if row["customer_name"] is not None:
        customer = Customer(
            name=str(row["customer_name"]),
            email=str(row["email"]),
            city=str(row["city"]),
        )
    return Order(
        number=str(row["number"]),
        status=str(row["status"]),
        placed_on=date.fromisoformat(str(row["placed_on"])),
        promised_on=_date(row["promised_on"]),
        delivered_on=_date(row["delivered_on"]),
        customer=customer,
        lines=lines,
    )


def _iso(value: date | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _date(value: object) -> date | None:
    if value is None or value == "":
        return None
    return date.fromisoformat(str(value))
