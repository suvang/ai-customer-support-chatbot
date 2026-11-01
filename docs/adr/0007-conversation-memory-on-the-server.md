# Conversation memory on the server

A Customer asks follow-ups such as "what did he order last?" or "can I get a refund for it?", and each message used to be answered on its own. The server now keeps a short Conversation per `conversation_id`: the last 6 turns (the Customer's words and the reply text, capped at 1,200 characters) and the Order numbers and Customers the last record-bearing reply was about.

- **AI mode** replays those turns as chat messages and adds one line naming the records in focus. Tool payloads are not replayed: the model calls a tool again for fresh facts, so a remembered turn never becomes a source of Order facts (ADR 0005), and the prompt stays small.
- **Offline mode** resolves "he", "she", "it", "that order" to the records in focus, and reads "last" as the most recently placed of the matched Orders.

The store is in memory, bounded (500 Conversations, least recently used dropped first, 2 hours idle timeout), and cleared on restart. We rejected sending the whole history from the browser on every request, because it grows each turn and would let the client rewrite what the assistant "said". We also rejected summarising history with an extra model call, which costs a request per turn for a few short turns. A database-backed store can replace `ConversationStore` later without changing callers.
