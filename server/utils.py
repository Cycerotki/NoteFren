from pydantic import BaseModel
from typing import List

class MessageReq(BaseModel):
    message: str

class Response(BaseModel):
    code: int = 200

class TextResp(Response):
    text: str

class ListResp(Response):
    data: List[str]