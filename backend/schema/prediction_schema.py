from pydantic import BaseModel
from typing import Dict

class Employee(BaseModel):
    features: Dict[str, float]