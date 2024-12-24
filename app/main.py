from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.routes import root
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from app.routes.knn import knn_v1
from app.routes.catboost import catboost_v1

app = FastAPI()

# Exception Handler untuk RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = exc.body.decode("utf-8") if isinstance(exc.body, bytes) else exc.body
    errors = [
        {
            "field": ".".join(map(str, error["loc"])),
            "error_message": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"errors": errors, "body_received": body},
    )

logger = logging.getLogger("uvicorn.error")

class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        body = await request.body()
        logger.info(f"Request body: {body.decode('utf-8')}")
        response = await call_next(request)
        return response

app.add_middleware(LogRequestMiddleware)

# Include routes
app.include_router(root.router, tags=["Root"])
app.include_router(knn_v1.router, prefix="/api/knn/v1", tags=["Prediction KNN v0.1"])
app.include_router(catboost_v1.router, prefix="/api/catboost/v1", tags=["Prediction Catboost v0.1"])