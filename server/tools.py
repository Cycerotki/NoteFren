from typing import Tuple, Dict

import easyocr
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

# global objects
SEARCH_ENGINE = DuckDuckGoSearchRun()
READER = easyocr.Reader(['ch_sim','en'])
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


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

@tool
def wikipedia(query: str) -> str:
    """
    Search Wikipedia on a given term

    Args:
        query (str): search term

    Returns:
        str: Relevant Wikipedia page
    """
    return wikipedia.run(query)

def text_from_image(data: bytes) -> str:
    """
    Performs Optical Character Recognition (OCR) on a given image and extracts text

    Args:
        data (bytes): file path of image
    
    Returns:
        str: extracted text
    """
    return READER.readtext(data, detail = 0)

def tool_runner_init() -> Tuple[ChatOllama, ChatOllama, Dict[str, StructuredTool]]:
    tools = {"search": search, 'wikipedia': wikipedia}
    tool_llm = ChatOllama(model = "llama3.1").bind_tools(list(tools.values()))
    summariser = ChatOllama(model = 'llama3.1')
    return tool_llm, summariser, tools