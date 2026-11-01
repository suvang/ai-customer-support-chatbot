"""The assistant answers from Support documents, an Order record, or both."""

from datetime import date, timedelta

from fastapi.testclient import TestClient

from store_support.agent import ModelTurn, ToolCall, ToolSession
from store_support.api import create_app
from store_support.assistant import SupportAssistant
from store_support.factory import build_assistant, load_sample_documents
from store_support.library import HashEmbedder, SupportLibrary
from store_support.orders import OrderBook, seed_demo_orders
from store_support.reply import LocalReplyWriter
from store_support.types import Order

TODAY = date(2026, 9, 26)


class ScriptedSession(ToolSession):
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = turns
        self._index = 0
        self.user = ""
        self.results: list[list[tuple[ToolCall, str]]] = []

    def start(self, system: str, user: str, tools: list[dict[str, object]], history=()) -> ModelTurn:
        self.system = system
        self.user = user
        self.history = history
        return self._next()

    def resume(self, results: list[tuple[ToolCall, str]]) -> ModelTurn:
        self.results.append(results)
        return self._next()

    def _next(self) -> ModelTurn:
        turn = self._turns[self._index]
        self._index += 1
        return turn


def demo_assistant(writer=None) -> SupportAssistant:
    library = SupportLibrary(HashEmbedder())
    load_sample_documents(library)
    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    return SupportAssistant(library, orders, writer or LocalReplyWriter(), clock=lambda: TODAY)


def test_policy_question_cites_the_support_document_and_does_not_invent_an_order():
    reply = demo_assistant().ask("What is the refund policy?", today=TODAY)

    assert reply.escalation is False
    assert reply.order is None
    assert any(citation.document == "refund-policy.md" for citation in reply.citations)
    assert "30 days" in reply.text
    assert "Order 1048 is" not in reply.text


def test_unknown_question_escalates_without_inventing_policy():
    reply = demo_assistant().ask("What is the wifi password for the warehouse?", today=TODAY)

    assert reply.escalation is True
    assert reply.citations == ()
    assert reply.order is None
    assert "person on the Store support team" in reply.text
    assert "30 days" not in reply.text
    assert "shipped" not in reply.text.lower()


def test_known_order_uses_the_order_record():
    reply = demo_assistant().ask("Where is order 1042?", today=TODAY)

    assert reply.order is not None
    assert reply.order.number == "1042"
    assert reply.order.status == "shipped"
    assert "Order 1042 is shipped." in reply.text
    assert reply.order.promised_on is not None
    assert reply.order.promised_on.isoformat() in reply.text
    assert "It has not been delivered." in reply.text
    assert "delivered on" not in reply.text


def test_missing_order_does_not_invent_a_status():
    reply = demo_assistant().ask("Where is order 9999?", today=TODAY)

    assert reply.order is None
    assert "Order 9999 is not in the Store's records." in reply.text
    assert "I can't state an Order status" in reply.text
    assert "Order 9999 is shipped" not in reply.text
    assert "Order 9999 is delivered" not in reply.text
    assert reply.citations == ()


def test_refund_eligibility_follows_the_retrieved_window_and_the_order_record():
    library = SupportLibrary(HashEmbedder())
    library.add_document(
        "refund-policy.md",
        "A delivered Order can be refunded for 10 days after the delivery date. "
        "A cancelled Order cannot be refunded. "
        "An Order that has not been delivered cannot be refunded yet.",
    )
    orders = OrderBook()
    orders.add(Order("2001", "delivered", TODAY - timedelta(days=20), TODAY - timedelta(days=8), TODAY - timedelta(days=5)))
    orders.add(Order("2002", "delivered", TODAY - timedelta(days=40), TODAY - timedelta(days=30), TODAY - timedelta(days=15)))
    orders.add(Order("2003", "cancelled", TODAY - timedelta(days=4), TODAY - timedelta(days=3), None))
    assistant = SupportAssistant(library, orders, LocalReplyWriter())

    available = assistant.ask("Can I get a refund for order 2001?", today=TODAY)
    expired = assistant.ask("Can I get a refund for order 2002?", today=TODAY)
    cancelled = assistant.ask("Can I get a refund for order 2003?", today=TODAY)

    assert "10 days" in available.text
    assert "a Refund is available." in available.text
    assert "30 days" not in available.text
    assert "a Refund is not available." in expired.text
    assert "cannot be refunded" in cancelled.text


def test_each_store_table_can_be_read_on_its_own():
    from store_support.tools import StoreTools

    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    tools = StoreTools(orders, SupportLibrary(HashEmbedder()))

    customers = tools.execute("find_customers", {"city": "Brooklyn"})
    products = tools.execute("find_products", {"name": "kettle"})
    packed = tools.execute("find_orders", {"status": "packed"})
    prices = tools.execute("find_order_lines", {"product": "Night Market Candle"})

    assert customers.grounded is True
    assert customers.payload["customers"] == [
        {"name": "Samir Shah", "email": "samir.shah@example.com", "city": "Brooklyn"}
    ]
    assert products.payload["products"] == [
        {
            "sku": "KETTLE",
            "name": "Cedar Pour-Over Kettle",
            "category": "Kitchen",
            "description": "A stainless kettle with a gooseneck spout.",
        }
    ]
    assert [order.number for order in packed.orders] == ["1060"]
    lines = prices.payload["lines"]
    assert isinstance(lines, list)
    assert {line["order_number"] for line in lines} == {"1001", "1072"}
    assert {line["unit_price_cents"] for line in lines} == {3200}


def test_product_tool_call_answers_without_an_order():
    from store_support.agent import ModelTurn, ToolAgent, ToolCall
    from store_support.tools import StoreTools

    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    library = SupportLibrary(HashEmbedder())
    session = ScriptedSession(
        [
            ModelTurn("", (ToolCall("call-1", "find_products", {"name": "kettle"}),)),
            ModelTurn("The Cedar Pour-Over Kettle is a stainless kettle with a gooseneck spout.", ()),
        ]
    )
    assistant = SupportAssistant(
        library,
        orders,
        LocalReplyWriter(),
        agent=ToolAgent(lambda: session, StoreTools(orders, library)),
    )

    reply = assistant.ask("What is the Cedar Pour-Over Kettle?", today=TODAY)

    assert "gooseneck spout" in reply.text
    assert "gooseneck spout" in session.results[0][0][1]
    assert reply.orders == ()
    assert reply.escalation is False


def test_find_orders_matches_a_customer_name_fragment():
    from store_support.tools import StoreTools

    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    tools = StoreTools(orders, SupportLibrary(HashEmbedder()))

    outcome = tools.execute("find_orders", {"customer": "samir", "product": ""})

    assert [order.number for order in outcome.orders] == ["1048"]
    assert outcome.orders[0].customer is not None
    assert outcome.orders[0].customer.name == "Samir Shah"


def test_tool_call_loads_only_the_order_the_model_asks_for():
    from store_support.agent import ModelTurn, ToolAgent, ToolCall
    from store_support.tools import StoreTools

    library = SupportLibrary(HashEmbedder())
    load_sample_documents(library)
    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    session = ScriptedSession(
        [
            ModelTurn(
                "",
                (
                    ToolCall("call-1", "get_order", {"number": "1048"}),
                    ToolCall("call-2", "search_support_documents", {"query": "refund policy"}),
                ),
            ),
            ModelTurn("Samir Shah ordered an Oat Linen Throw.", ()),
        ]
    )
    assistant = SupportAssistant(
        library,
        orders,
        LocalReplyWriter(),
        agent=ToolAgent(lambda: session, StoreTools(orders, library)),
    )

    reply = assistant.ask("whats there in order number 1048?", today=TODAY)

    assert reply.text == "Samir Shah ordered an Oat Linen Throw."
    assert reply.order is not None
    assert reply.order.number == "1048"
    tool_payload = session.results[0][0][1] + session.results[0][1][1]
    assert "Samir Shah" in tool_payload
    assert "Oat Linen Throw" in tool_payload
    assert "1030" not in tool_payload
    assert "refund-policy.md" in tool_payload
    assert "order number 1048" in session.user


def test_empty_message_is_rejected():
    client = TestClient(create_app(demo_assistant()))

    response = client.post("/messages", json={"message": "   "})

    assert response.status_code == 400


def test_messages_endpoint_returns_order_facts_and_citations():
    client = TestClient(create_app(demo_assistant()))

    response = client.post("/messages", json={"message": "Can I get a refund?", "order_number": "1048"})

    assert response.status_code == 200
    body = response.json()
    assert body["escalation"] is False
    assert body["order"]["number"] == "1048"
    assert body["order"]["status"] == "delivered"
    assert any(citation["document"] == "refund-policy.md" for citation in body["citations"])
    assert "a Refund is available." in body["reply"]


def test_order_number_wording_finds_the_same_order_as_the_customer_name():
    assistant = demo_assistant()

    by_name = assistant.ask("what did Samir order?", today=TODAY)
    by_number = assistant.ask("whats there in order number 1048?", today=TODAY)

    assert by_name.order is None
    assert [order.number for order in by_name.orders] == ["1048"]
    assert by_number.order is not None
    assert by_number.order.number == "1048"
    for reply in (by_name, by_number):
        assert "Samir Shah" in reply.text
        assert "Field Notes Notebook" in reply.text
        assert "Oat Linen Throw" in reply.text


def test_order_contents_include_the_products_and_the_customer():
    reply = demo_assistant().ask("What is in order 1042?", today=TODAY)

    assert "Cedar Pour-Over Kettle" in reply.text
    assert "Glass Carafe" in reply.text
    assert "Adele Voss" in reply.text
    assert "$64.00" in reply.text
    assert "Jonah Park" not in reply.text


def test_product_question_lists_every_customer_who_ordered_it():
    reply = demo_assistant().ask("Which Customer ordered the Cedar Pour-Over Kettle?", today=TODAY)

    assert "Adele Voss" in reply.text
    assert "Mira Ellison" in reply.text
    assert "1042" in reply.text
    assert "1055" in reply.text
    assert "Jonah Park" not in reply.text


def test_customer_question_lists_that_customers_orders():
    reply = demo_assistant().ask("What did Mira Ellison order?", today=TODAY)

    assert "1001" in reply.text
    assert "Night Market Candle" in reply.text
    assert "1055" in reply.text
    assert "Travel Tumbler" in reply.text
    assert "Jonah Park" not in reply.text


def test_examples_are_questions_a_customer_can_ask():
    client = TestClient(create_app(demo_assistant()))

    response = client.get("/examples")

    assert response.status_code == 200
    messages = [item["message"] for item in response.json()["examples"]]
    assert "What is in order 1042?" in messages
    assert "Which Customer ordered the Cedar Pour-Over Kettle?" in messages


def test_factory_without_a_key_answers_from_the_local_library(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assistant = build_assistant(env={}, today=TODAY)

    reply = assistant.ask("What is a shipping delay?", today=TODAY)

    assert any(citation.document == "shipping-policy.md" for citation in reply.citations)
    assert reply.escalation is False


def test_reply_shows_the_steps_taken_and_a_delay_verdict():
    client = TestClient(create_app(demo_assistant()))

    body = client.post("/messages", json={"message": "Is order 1042 delayed?"}).json()

    assert "Order 1042 is delayed." in body["reply"]
    assert [step["label"] for step in body["steps"]] == [
        "Searched Order records",
        "Searched the Support library",
    ]
    assert "1042" in body["steps"][0]["detail"]


def test_greeting_explains_what_the_assistant_can_do_without_escalating():
    reply = demo_assistant().ask("hi", today=TODAY)

    assert reply.escalation is False
    assert "refund" in reply.text.lower()


def test_city_and_all_orders_questions_are_answered_from_the_records():
    assistant = demo_assistant()

    brooklyn = assistant.ask("Which Customer lives in Brooklyn?", today=TODAY)
    everything = assistant.ask("Show me all orders", today=TODAY)

    assert brooklyn.escalation is False
    assert "Samir Shah" in brooklyn.text
    assert len(everything.orders) == 7


def test_status_and_documents_describe_the_running_assistant():
    client = TestClient(create_app(demo_assistant()))

    assert client.get("/status").json() == {"mode": "local", "model": None}
    names = [document["name"] for document in client.get("/documents").json()["documents"]]
    assert "refund-policy.md" in names


def test_tool_agent_reports_each_tool_call_as_a_step():
    from store_support.agent import ToolAgent
    from store_support.tools import StoreTools

    library = SupportLibrary(HashEmbedder())
    load_sample_documents(library)
    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    session = ScriptedSession(
        [
            ModelTurn("", (ToolCall("call-1", "get_order", {"number": "1048"}),)),
            ModelTurn("Order 1048 is delivered.", ()),
        ]
    )
    assistant = SupportAssistant(
        library, orders, LocalReplyWriter(), agent=ToolAgent(lambda: session, StoreTools(orders, library))
    )

    reply = assistant.ask("where is order 1048?", today=TODAY)

    assert reply.steps[0].label == "Opened an Order record"
    assert reply.steps[0].detail == "Order 1048: found"


def test_follow_up_question_remembers_the_customer_from_the_last_reply():
    client = TestClient(create_app(demo_assistant()))

    first = client.post("/messages", json={"message": "show me the list of customers living in brooklyn?"}).json()
    second = client.post(
        "/messages",
        json={"message": "what did he order last?", "conversation_id": first["conversation_id"]},
    ).json()

    assert first["conversation_id"]
    assert second["conversation_id"] == first["conversation_id"]
    assert second["escalation"] is False
    assert [order["number"] for order in second["orders"]] == ["1048"]
    assert "Oat Linen Throw" in second["reply"]
    assert second["steps"][0]["label"] == "Used conversation memory"


def test_last_picks_the_most_recent_of_a_customers_orders():
    assistant = demo_assistant()
    first = assistant.ask("What did Mira Ellison order?", today=TODAY)

    latest = assistant.ask("what did she order last?", today=TODAY, conversation_id=first.conversation_id)

    assert latest.order is not None
    assert latest.order.number == "1001"


def test_it_refers_to_the_order_discussed_before():
    assistant = demo_assistant()
    first = assistant.ask("What is in order 1048?", today=TODAY)

    refund = assistant.ask("Can I get a refund for it?", today=TODAY, conversation_id=first.conversation_id)

    assert refund.order is not None
    assert refund.order.number == "1048"
    assert "a Refund is available." in refund.text


def test_a_new_conversation_does_not_inherit_another_ones_memory():
    assistant = demo_assistant()
    assistant.ask("What is in order 1048?", today=TODAY)

    fresh = assistant.ask("Can I get a refund for it?", today=TODAY)

    assert fresh.order is None


def test_tool_agent_receives_earlier_turns_and_the_records_in_focus():
    from store_support.agent import ToolAgent
    from store_support.tools import StoreTools

    library = SupportLibrary(HashEmbedder())
    orders = OrderBook()
    seed_demo_orders(orders, TODAY)
    sessions: list[ScriptedSession] = []

    def factory() -> ScriptedSession:
        script = [
            [
                ModelTurn("", (ToolCall("c1", "find_customers", {"city": "Brooklyn"}),)),
                ModelTurn("Samir Shah lives in Brooklyn.", ()),
            ],
            [
                ModelTurn("", (ToolCall("c2", "find_orders", {"customer": "Samir Shah"}),)),
                ModelTurn("Samir Shah last ordered order 1048.", ()),
            ],
        ][len(sessions)]
        sessions.append(ScriptedSession(script))
        return sessions[-1]

    assistant = SupportAssistant(
        library, orders, LocalReplyWriter(), agent=ToolAgent(factory, StoreTools(orders, library))
    )

    first = assistant.ask("show me the customers living in brooklyn", today=TODAY)
    assistant.ask("what did he order last?", today=TODAY, conversation_id=first.conversation_id)

    assert [customer.name for customer in first.customers] == ["Samir Shah"]
    second = sessions[1]
    assert [turn.user for turn in second.history] == ["show me the customers living in brooklyn"]
    assert second.history[0].assistant == "Samir Shah lives in Brooklyn."
    assert "Samir Shah" in second.system


def test_conversation_store_is_bounded():
    from store_support.memory import MAX_TURNS, ConversationStore
    from store_support.types import Reply

    store = ConversationStore(max_conversations=2)
    first = store.open(None)
    for index in range(MAX_TURNS + 3):
        store.record(first, f"q{index}", Reply("a", (), None, (), False))
    store.open(None)
    store.open(None)

    assert len(first.history()) == MAX_TURNS
    assert len(store) == 2
    assert store.open(first.id).id != first.id
