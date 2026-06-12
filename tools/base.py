import functools
import logging
from typing import Callable

logger = logging.getLogger(__name__)

def handle_tool_error(func: Callable) -> Callable:
    """Decorator bắt mọi exception trong tool, trả về error message thay vì crash."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Tool '{func.__name__}' failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Sorry, I encountered an error while processing your request. Please try again."
    return wrapper