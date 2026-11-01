# Local Support library, OpenAI optional

The assistant must run with no API key: a local Support library returns the closest Support document as the Grounded answer. When `OPENAI_API_KEY` is set, retrieval and the reply use OpenAI. We rejected a Hugging Face fine-tune (see ADR 0001) and an OpenAI-only setup, which would not run in a demo without credentials.
