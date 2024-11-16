from typing import Tuple, Dict

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

# global objects
SEARCH_ENGINE = DuckDuckGoSearchRun()


@tool
def search(question: str) -> str:
    """
    Ask DuckDuckGo a question. Does not allow asynchronous calls

    Args:
        question (str): The question to ask.

    Returns:
        str: Top search result.
    """
    print(question)
    return SEARCH_ENGINE.invoke(question)

def tool_runner_init() -> Tuple[ChatOllama, ChatOllama, Dict[str, StructuredTool]]:
    tools = {"search": search}
    tool_llm = ChatOllama(model = "llama3.1").bind_tools(list(tools.values()))
    summariser = ChatOllama(model = 'llama3.1')
    return tool_llm, summariser, tools