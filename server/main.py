# standard library
from datetime import datetime
import os
from typing import Dict, List

# external libraries
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from langchain_core.messages import ToolMessage
import uvicorn

# user defined
from .prompts import *
from .tools import *
from .utils import *



# FastAPI runs on port 8000
app = FastAPI()
agent_llm, content_llm, TOOLS = tool_runner_init()




@app.get('/')
async def root() -> Dict[str, str]:
    return {'message': 'Hello World!'}

@app.post('/sampleWiki/')
async def sample_wiki(req: MessageReq) -> TextResp:
    info = wikipedia(req.message)
    content = f'**Search Term: {req.message}**\n\n{info}\n'
    if os.path.exists(f'logs/log_wiki_{datetime.now().strftime("%Y_%m_%d")}.txt'):
        content = '\n\n' + content
        
    with open(f'logs/log_wiki_{datetime.now().strftime("%Y_%m_%d")}.txt', 'a') as f:
        f.write(content)
    summary = content_llm.invoke(summary_prompt(info))
    
    return TextResp(text=f'Related content: {summary.content}')

@app.post('/multicall/')
async def multi_call(req: MessageReq) -> Dict[str, List[str]]:
    res = []
    for tool_call in agent_llm.invoke(multi_call_prompt(req.message)).tool_calls:
        selected_tool = TOOLS[tool_call["name"].lower()]
        res.append(
            ToolMessage(
                selected_tool.invoke(tool_call['args']), tool_call_id=tool_call['id']
            ).content
        )
        
    return {'result': res}

@app.post('/ask/')
async def ask(req: MessageReq) -> TextResp:
    tool_call = agent_llm.invoke(search_prompt(req.message)).tool_calls[0]
    print(tool_call)
    info = TOOLS[tool_call["name"].lower()].invoke(tool_call['args'])
    res = content_llm.invoke(q_n_a_prompt(req.message, info))
    
    return TextResp(text=res.content)

@app.post('/ocr/')
async def ocr(image: UploadFile) -> ListResp:
    return ListResp(data=text_from_image(await image.read()))

@app.post('/asr/')
async def asr(audio: UploadFile) -> ListResp:
    return ListResp(data=transcribe(audio))

@app.post('/podcast/')
async def podcast(req: MessageReq) -> FileResponse:
    topic = req.message
    transcript_text = content_llm.invoke(generate_podcast_prompt(topic)).content
    transcript = []
    for row in transcript_text.split('\n'):
        if row and row[0:2] in ('1:', '2:'):
            transcript.append([int(row[0])-1, row[3:]])

    return FileResponse(generate_podcast(transcript))

@app.post('/pdf/')
async def pdf(pdf: UploadFile) -> TextResp:
    with open(f'assets/{pdf.filename}', 'wb') as f:
        f.write(pdf.file.read())
    content, is_valid = parse_file(f'assets/{pdf.filename}')
    if is_valid:
        ls = content['content']
        for i, pg in enumerate(ls):
            ls[i] = content_llm.invoke(summary_prompt(pg))
    content['content'] = content_llm.invoke(list_summary_prompt(', '.join(ls)))
    return TextResp(text=content['content'])


if __name__=='__main__':
    # not necessary unless hostname or port needs to change
    # to run in dev, use fastapi dev
    # to specify server file, use fastapi dev <filename.py>
    # for production, use fastapi run
    uvicorn.run(app, host="localhost", port=8000)