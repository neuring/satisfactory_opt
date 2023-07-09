from error import Error
from source import Span, locate


def report_error(error: Error, input_str: str):

    print(f"ERROR: {error.create_message()}")
    print()

    for span in error.span:
        _report_content(span, input_str)

    print()


def _report_content(span: Span, input_str: str):
    location = locate(span, input_str)

    content = location.content.strip().splitlines()

    print(f"At {location.line}:{location.column}")

    for idx, content_line in enumerate(content):
        if idx == 0:
            padding = " " * location.column
        else:
            padding = ""
        
        print(f"{location.line + idx}\t: {padding}{content_line}")
