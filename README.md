# Store support

A Customer can ask about Store policy, about one Order, and about who bought a Product. With `OPENAI_API_KEY` set, the model chooses a tool for the customers, products, orders, or order_lines table, or searches Support documents, and the app runs that query. Without a key, the same questions are answered locally from the records and the Support library. The reply does not invent a fact the tool did not return.

Open http://127.0.0.1:8000 after the server starts.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn store_support.asgi:app --app-dir src --port 8000
```

Without `OPENAI_API_KEY`, retrieval and the reply stay local. With a key, both use OpenAI. Restart the server after changing `.env`.

## Using the app

- **Left panel: Sample store data.** Every Order, Customer, and Policy the assistant can see. Click an Order for quick questions ("What's in it?", "Is it late?", "Refund?"), a Customer to see what they ordered, or a Policy to read it.
- **Chat.** Pick a suggested question or type your own (Enter sends, Shift+Enter adds a line).
- **Each reply shows** the Order cards it used, the policy **Sources** it quoted (click to read the passage), and **How I found this**: every lookup the assistant ran.
- **Mode badge** (top right): *AI mode* when `OPENAI_API_KEY` is set, *Offline mode* otherwise. Hover it for details.
- If nothing in the records or policies answers the question, the reply is marked **Not found** and suggests a person instead of guessing.
- **Memory.** Follow-ups work: ask "Which Customer lives in Brooklyn?" then "What did he order last?" or "Can I get a refund for it?". The **Remembering** bar above the message box shows the records in focus; **New chat** clears it. See `docs/adr/0007-conversation-memory-on-the-server.md`.

## API

| Method | Path | What it returns |
| --- | --- | --- |
| `POST` | `/messages` | `reply`, `orders`, `order`, `citations`, `steps`, `escalation`, `conversation_id`, `remembering`. Send `conversation_id` back to continue a conversation. |
| `DELETE` | `/conversations/{id}` | Forgets that conversation |
| `GET` | `/orders` | Every Order record |
| `GET` | `/documents` | Every Support document |
| `GET` | `/examples` | Suggested questions, grouped |
| `GET` | `/status` | `{"mode": "ai" \| "local", "model": ...}` |

## Try

Sample Customers, Products, and Orders are seeded when the server starts. The page lists questions you can ask, including:

- What is in order 1042?
- Which Customer ordered the Cedar Pour-Over Kettle?
- What did Mira Ellison order?
- Can I get a refund for order 1048?

```bash
curl -H "content-type: application/json" \
  -d '{"message":"Can I get a refund?","order_number":"1048"}' \
  http://127.0.0.1:8000/messages
```
