from fastapi import FastAPI
from typing import Dict

@app.get('/')
async def root() -> Dict[str, str]:
    return {'message': 'Hello World!'}

if __name__ == '__main__':
    # FastAPI runs on port 8000
    app = FastAPI()