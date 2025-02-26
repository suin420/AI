from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
def read_root(item: str):
    return {"result": "item"}
