# Store support

Language for helping a Customer of an online Store with Orders, Shipments, and Refunds.

## Language

### Store and Customer

**Store**:
The online retailer the Customer is contacting.
_Avoid_: Shop, merchant, company

**Customer**:
A person who places Orders with the Store.
_Avoid_: User, client, buyer, account

### Orders and Shipments

**Order**:
A purchase the Customer placed with the Store.
_Avoid_: Purchase, transaction, cart

**Order record**:
The Store's stored facts for one Order: its number, Order status, dates, the Customer who placed it, and its Order lines.
_Avoid_: Account, profile

**Product**:
A thing the Store sells, with a name, a price, and a short description.
_Avoid_: Item, SKU, goods

**Order line**:
One Product and a quantity on an Order.
_Avoid_: Line item, cart item

**Order status**:
Where an Order is in fulfillment: placed, packed, shipped, delivered, or cancelled.
_Avoid_: Tracking state, fulfillment state

**Shipment**:
The delivery of an Order to the Customer.
_Avoid_: Package, parcel, delivery

**Shipping delay**:
A Shipment that has not arrived by the date the Store promised.
_Avoid_: Late package, shipping issue

### Refunds

**Refund**:
Money the Store returns to the Customer for an eligible Order.
_Avoid_: Return, reimbursement, chargeback

**Refund eligibility**:
Whether Store policy allows a Refund for that Order.
_Avoid_: Return window, refund rules

### Support

**Support document**:
A Store policy or FAQ the assistant may quote.
_Avoid_: Article, knowledge article, doc

**Support library**:
The set of Support documents the assistant may search.
_Avoid_: Knowledge base, corpus, vector store

**Citation**:
The Support document passage a reply is based on.
_Avoid_: Source, reference, link

**Grounded answer**:
A reply that states only facts from Order records, retrieved Support document text, or both, and includes a Citation when it uses a Support document.
_Avoid_: Response, completion

**Escalation**:
Sending the Customer to a person when the assistant has neither a relevant Support document nor an Order record.
_Avoid_: Handoff, ticket, transfer

**Troubleshooting path**:
A short sequence of checks drawn from Support documents.
_Avoid_: Flow, workflow, runbook
