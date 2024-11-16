from pydantic import BaseModel

class MessageReq(BaseModel):
    message: str

class Response(BaseModel):
    code: int = 200

class TestResp(Response):
    text: str