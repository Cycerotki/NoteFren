from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama

SEARCH_ENGINE = DuckDuckGoSearchRun()

@tool
def search(msg: str) -> str:
    return SEARCH_ENGINE.invoke(msg)

def tool_runner() -> ChatOllama:
    tools = [search]
    tool_llm = ChatOllama(model = "qwen2.5").bind_tools(tools)
    return tool_llm