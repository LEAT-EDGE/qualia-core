import logging

class TextFormatter(logging.Formatter):
    def format(self, record):
        return record.msg
