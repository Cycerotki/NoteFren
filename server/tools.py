import subprocess, os
from typing import Tuple, Dict, List

import cv2
import easyocr
from fastapi import UploadFile
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
import numpy as np
import outetts
from outetts.version.v1.interface import ModelOutput
import torch

import pdfplumber
from rapid_latex_ocr import LatexOCR
from docling.document_converter import DocumentConverter


# global objects
SEARCH_ENGINE = DuckDuckGoSearchRun()
READER = easyocr.Reader(['en'])
DOCUMENT_CONVERTER = DocumentConverter()
LATEX_OCR = LatexOCR()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)
INTERFACE = outetts.InterfaceHF(model_version="0.2", cfg=model_config)


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
        List[str]: transcript
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

def generate_podcast(transcript: List[List[str]]):
    """
    Converts text into a 2-speaker podcast

    Args:
        content (List[List[str]]): podcast content
    
    Returns:
        AudioFile: generated torchaudio file
    """
    speakers = (INTERFACE.load_default_speaker(name='male_4'),
                INTERFACE.load_default_speaker(name='female_2'))
    audiofiles: List[ModelOutput] = []

    for i, content in transcript:
        audiofiles.append(
            INTERFACE.generate(
                text=content,
                # Lower temperature values may result in a more stable tone,
                # while higher values can introduce varied and expressive speech
                temperature=0.3,
                repetition_penalty=1.1,
                max_length=4096,
                speaker=speakers[i],
            )
        )

    output = ModelOutput(torch.cat([rec.audio for rec in audiofiles], dim=1), audiofiles[0].sr, audiofiles[0].enable_playback)
    # Save the synthesized speech to a file
    output.save(f"output.wav")
    return 'output.wav'

def parse_file(filepath: str, page: int = 1) -> Tuple[Dict[str, str], bool]:
    """
    Extracts text from file

    Args:
        filepath(str): server filepath of downloaded file

    Returns:
        str: text content of file
    """
    match (filepath.split('.')[-1].lower()):
        case 'txt':
            with open(filepath) as f:
                content = [''.join(f.readlines())]
            # os.remove(filepath)
            return {'content': content}, True
        case 'pdf':
            # initialization
            content = []
            latex = []
            result = DOCUMENT_CONVERTER.convert(filepath)
            markdown_result = result.document.export_to_markdown() 

            # read PDF file (load all pages in the PDF file)
            with pdfplumber.open(filepath) as pdf:
                page_count = len(pdf.pages)
                if page > page_count:
                    return [f'Page {page} does not exist in the PDF'], False
                for page_index in range(page_count):  # traverse all pages
                    page = pdf.pages[page_index]  # select the current page
                    page_image = page.to_image(resolution=150) # convert the page to image by default (20230815)
                    image = cv2.cvtColor(np.array(page_image.original), cv2.COLOR_RGB2BGR)

                    # PROCESS TWO ENGINES
                    ocr_result = READER.readtext(image, detail = 0)
                    latex_result, elapse = LATEX_OCR(image)

                    content.append(ocr_result)
                    latex.append(latex_result)            
        case _:
            return {"content": 'File not supported, send a text or pdf file', "latex": '', "markdown": ''}, False
    # delete file from server
    os.remove(filepath)
    return {"content": content, "latex": latex, "markdown": markdown_result}, True

def tool_runner_init() -> Tuple[ChatOllama, ChatOllama, Dict[str, StructuredTool]]:
    tools = {"search": search, 'wikipedia': wikipedia}
    tool_llm = ChatOllama(model = "llama3.1").bind_tools(list(tools.values()))
    summariser = ChatOllama(model = 'qwen2.5')
    return tool_llm, summariser, tools