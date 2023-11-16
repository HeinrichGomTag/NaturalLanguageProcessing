from typing import Union

from fastapi import FastAPI, Request, Query
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    message: str

@app.post("/response")
async def root(data: Item):
    respuestas = {"Saludar": "Hola", "Despedir": "Adios"}
    # print(str(data.message))
    return {"message": f"Tu respuesta: '{respuestas[str(data.message)]}'"}
    # return {}
