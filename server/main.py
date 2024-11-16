# standard library
from datetime import datetime
import os
from typing import Dict, List

# external libraries
from fastapi import FastAPI
from langchain_core.messages import ToolMessage
import uvicorn

# user defined
from prompts import *
from tools import *
from utils import *



# FastAPI runs on port 8000
app = FastAPI()
llm, summariser, TOOLS = tool_runner_init()




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
    summary = summariser.invoke(summary_prompt(info))
    
    return TextResp(text=f'Related content: {summary.content}')

@app.post('/multicall/')
async def multi_call(req: MessageReq) -> Dict[str, List[str]]:
    res = []
    for tool_call in llm.invoke(multi_call_prompt(req.message)).tool_calls:
        selected_tool = TOOLS[tool_call["name"].lower()]
        res.append(ToolMessage(
            selected_tool.invoke(tool_call['args']), tool_call_id=tool_call['id']
            ).content)
        
    return {'result': res}

@app.post('/ask/')
async def ask(req: MessageReq) -> TextResp:
    tool_call = llm.invoke(search_prompt(req.message)).tool_calls[0]
    print(tool_call)
    info = TOOLS[tool_call["name"].lower()].invoke(tool_call['args'])
    res = summariser.invoke(q_n_a_prompt(req.message, info))
    
    return TextResp(text=res.content)



if __name__=='__main__':
    # not necessary unless hostname or port needs to change
    # to run in dev, use fastapi dev
    # to specify server file, use fastapi dev <filename.py>
    # for production, use fastapi run
    uvicorn.run(app, host="localhost", port=8000)