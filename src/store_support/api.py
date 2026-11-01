from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from store_support.assistant import SupportAssistant
from store_support.factory import build_assistant
from store_support.types import Order, Reply

PAGE = Path(__file__).parent / "static" / "index.html"


class MessageBody(BaseModel):
    message: str
    order_number: str | None = None
    conversation_id: str | None = None


def create_app(assistant: SupportAssistant | None = None) -> FastAPI:
    app = FastAPI(title="Store support")
    service = assistant or build_assistant()

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, str | None]:
        return service.status()

    @app.get("/documents")
    def documents() -> dict[str, list[dict[str, str]]]:
        return {"documents": list(service.documents())}

    @app.get("/orders")
    def orders() -> dict[str, list[dict[str, object]]]:
        return {"orders": [_order_json(order) for order in service.orders()]}

    @app.get("/examples")
    def examples() -> dict[str, list[dict[str, str | None]]]:
        return {"examples": list(service.examples())}

    @app.post("/messages")
    def messages(body: MessageBody) -> dict[str, object]:
        try:
            reply = service.ask(
                body.message,
                order_number=body.order_number,
                conversation_id=body.conversation_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        body_json = _reply_json(reply)
        body_json["remembering"] = service.remembered(reply.conversation_id or "")
        return body_json

    @app.delete("/conversations/{conversation_id}")
    def forget(conversation_id: str) -> dict[str, bool]:
        service.forget(conversation_id)
        return {"forgotten": True}

    return app


def _reply_json(reply: Reply) -> dict[str, object]:
    return {
        "reply": reply.text,
        "conversation_id": reply.conversation_id,
        "escalation": reply.escalation,
        "order": _order_json(reply.order) if reply.order else None,
        "orders": [_order_json(order) for order in reply.orders],
        "citations": [
            {"document": citation.document, "excerpt": citation.excerpt}
            for citation in reply.citations
        ],
        "steps": [{"label": step.label, "detail": step.detail} for step in reply.steps],
    }


def _order_json(order: Order) -> dict[str, object]:
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
                "name": line.product.name,
                "quantity": line.quantity,
                "unit_price_cents": line.unit_price_cents,
            }
            for line in order.lines
        ],
    }
