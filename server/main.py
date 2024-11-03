from fastapi import FastAPI
from typing import Dict

from tools import tool_runner

# FastAPI runs on port 8000
app = FastAPI()


@app.get('/')
async def root() -> Dict[str, str]:
    return {'message': 'Hello World!'}

@app.get('/cmd')
async def command(msg: str) -> Dict[str, any]:
    model = tool_runner()
    res = model.invoke(msg)
    return res

if __name__ == '__main__':
    app.run()