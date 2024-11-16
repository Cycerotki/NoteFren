import subprocess, os
from typing import Tuple, Dict, List

import easyocr
from fastapi import UploadFile
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

# global objects
SEARCH_ENGINE = DuckDuckGoSearchRun()
READER = easyocr.Reader(['en'])
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

def text_from_image(data: bytes) -> List[str]:
    """
    Performs Optical Character Recognition (OCR) on a given image and extracts text

    Args:
        data (bytes): file path of image
    
    Returns:
        List[str]: extracted text
    """
    return READER.readtext(data, detail = 0)

def transcribe(audio: UploadFile) -> List[str]:
    """
    Performs Speech-to-text/automatic speech recognition on audio file and transcribes

    Args:
        audio (UploadFile): audio file
    
    Returns:
        list[str]: transcript
    """
    path = '../whisper.cpp'
    name = 'assets/'+audio.filename.split('.')[0]

    with open(f'assets/{audio.filename}', 'wb') as f:
        f.write(audio.file.read())
    if 'wav' not in audio.filename:
        subprocess.run(['ffmpeg', '-y', '-i', f'assets/{audio.filename}', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', f'{name}.wav'])
        os.remove(f'assets/{audio.filename}')

    subprocess.run([f'{path}/main', '-m', f'{path}/models/ggml-medium.bin', '-f', f'{name}.wav', '--output-txt', 'true', '--output-file', f'{name}'])
    with open(f'{name}.txt') as f:
        text = [line.strip('\ "\n') for line in f.readlines()]
    os.remove(f'{name}.txt')
    os.remove(f'{name}.wav')
    return text


def tool_runner_init() -> Tuple[ChatOllama, ChatOllama, Dict[str, StructuredTool]]:
    tools = {"search": search, 'wikipedia': wikipedia}
    tool_llm = ChatOllama(model = "llama3.1").bind_tools(list(tools.values()))
    summariser = ChatOllama(model = 'llama3.1')
    return tool_llm, summariser, tools