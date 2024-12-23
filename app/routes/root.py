from fastapi import APIRouter

router = APIRouter()
  
@router.get("/")
def root():
    return {"version": "1.0"}

@router.get("/health")
def health():
    return {"status": "up"}
