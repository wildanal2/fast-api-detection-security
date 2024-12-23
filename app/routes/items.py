from fastapi import APIRouter, HTTPException
from app.models.form_req import FormReq

router = APIRouter()

items = []  # In-memory storage (can be replaced with a database)

@router.get("/items/{item_id}", response_model=FormReq)
def get_item(item_id: int):
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
