from pydantic import BaseModel
from typing import Optional

class Question(BaseModel):
    query:str
    api_key: Optional[str] = None