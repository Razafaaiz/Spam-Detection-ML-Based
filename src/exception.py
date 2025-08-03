import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    try:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        error_message = (
            f"Error occurred in Python script: [{file_name}] "
            f"at line number [{line_number}] "
            f"with error message: [{str(error)}]"
        )
        return error_message
    except Exception as inner_error:
        return f"Failed to extract error details due to: {str(inner_error)}"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
