import os
from datetime import datetime
from typing import Any
from pytz import timezone

def get_curr_time() -> str:
    # get current date and time in PST as str.
    return datetime.now().astimezone(
        timezone('Europe/Berlin')
    ).strftime("%d/%m/%Y %H:%M:%S")

# to check
class Logger: 
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if output_dir is not None and filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool = True) -> None:
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'): 
                message = '\n' + get_curr_time() + ' >> ' + message[1:]
            else:
                message = get_curr_time() + ' >> ' + message
        print(message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message + '\n')