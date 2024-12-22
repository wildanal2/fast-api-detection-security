from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text: str = None
    is_done: bool = False


items = []

@app.get("/")
def root():
    return {"hello": "word"}

@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return items

@app.get("/items", response_model=list[Item])
def list_item(limit: int=10):
    return items[0:limit]

@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int) -> str:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"items {item_id} tidak ditemukan")

