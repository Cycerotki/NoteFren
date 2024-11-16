from fastapi import FastAPI
from typing import Dict, List
import uvicorn

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage

# from prompts import *
# from tools import *
# from utils import *

# FastAPI runs on port 8000
app: FastAPI = FastAPI()
llm, summariser, TOOLS = tool_runner_init()


@app.get('/')
async def root() -> Dict[str, str]:
    return {'message': 'Hello World!'}

@app.post('/sample/')
async def sample(req: MessageReq) -> TestResp:
    print(req.message)
    return TestResp(text=f'Hi, you said {req.message}')

@app.post('/multicall/')
async def multi_call(req: MessageReq):
    res = []
    for tool_call in llm.invoke(multi_call_prompt(req.message)).tool_calls:
        selected_tool = TOOLS[tool_call["name"].lower()]
        res.append(ToolMessage(
            selected_tool.invoke(tool_call['args']), tool_call_id=tool_call['id']
            ).content)
    return {'result': res}

@app.post('/ask/')
async def ask(req: MessageReq):
    tool_call = llm.invoke(search_prompt(req.message)).tool_calls[0]
    print(tool_call)
    info = TOOLS[tool_call["name"].lower()].invoke(tool_call['args'])
    res = summariser.invoke(summarise_prompt(req.message, info))
    return {'result': res.content}



if __name__=='__main__':
    # not necessary unless hostname or port needs to change
    # to run in dev, use fastapi dev
    # to specify server file, use fastapi dev <filename.py>
    # for production, use fastapi run
    uvicorn.run(app, host="localhost", port=8000)