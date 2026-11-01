import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from store_support.agent import LangChainToolSession, ToolAgent
from store_support.assistant import SupportAssistant
from store_support.library import HashEmbedder, OpenAIEmbedder, SupportLibrary
from store_support.orders import OrderBook, seed_demo_orders
from store_support.reply import LocalReplyWriter
from store_support.tools import StoreTools

SAMPLE_DOCS = Path(__file__).parent / "sample_docs"
CHAT_MODEL = "gpt-4o-mini"


def build_assistant(
    env: Mapping[str, str] | None = None,
    today: date | None = None,
) -> SupportAssistant:
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        settings: Mapping[str, str] = os.environ
    else:
        settings = env
    openai_key = settings.get("OPENAI_API_KEY") or ""
    agent: ToolAgent | None = None
    writer = LocalReplyWriter()
    if openai_key:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        embedder: HashEmbedder | OpenAIEmbedder = OpenAIEmbedder(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)  # type: ignore[arg-type]
        )
        model = ChatOpenAI(model=CHAT_MODEL, api_key=openai_key)  # type: ignore[arg-type]
        library = SupportLibrary(embedder)
        load_sample_documents(library)
        orders = OrderBook()
        seed_demo_orders(orders, today or date.today())
        tools = StoreTools(orders, library)
        agent = ToolAgent(lambda: LangChainToolSession(model), tools)
        return SupportAssistant(library, orders, writer, agent=agent, model_name=CHAT_MODEL)
    embedder = HashEmbedder()
    library = SupportLibrary(embedder)
    load_sample_documents(library)
    orders = OrderBook()
    seed_demo_orders(orders, today or date.today())
    return SupportAssistant(library, orders, writer)


def load_sample_documents(library: SupportLibrary) -> None:
    for path in sorted(SAMPLE_DOCS.glob("*.md")):
        library.add_document(path.name, path.read_text(encoding="utf-8"))
