"""Централизованная обработка ошибок."""
from litestar import Request, Response, MediaType
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from api.logger import get_logger

logger = get_logger("exceptions")

def http_exception_handler(req: Request, exc: HTTPException) -> Response:
    logger.warning(f"{req.method} {req.url.path} → {exc.status_code}: {exc.detail}")
    return Response(
        content={"error": exc.detail, "status_code": exc.status_code},
        status_code=exc.status_code,
        media_type=MediaType.JSON,
    )

def internal_exception_handler(req: Request, exc: Exception) -> Response:
    logger.error(f"{req.method} {req.url.path} → 500: {exc}", exc_info=True)
    return Response(
        content={"error": "Внутренняя ошибка сервера", "status_code": 500},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        media_type=MediaType.JSON,
    )
