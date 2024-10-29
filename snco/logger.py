import logging
from datetime import datetime

import click


LOG_FORMATTING = {
    'INFO': dict(fg='green', label='INFO'),
    'ERROR': dict(fg='red', label='ERROR'),
    'EXCEPTION': dict(fg='red', label='EXCPT'),
    'CRITICAL': dict(fg='red', label='CRIT'),
    'DEBUG': dict(fg='blue', label='DEBUG'),
    'WARNING': dict(fg='yellow', label='WARN')
}


def format_log_msg(lvl, msg):
    dt = datetime.now().strftime("%Y%m%d:%H:%M:%S")
    fmt = LOG_FORMATTING[lvl]
    label = fmt['label'].rjust(5)
    prefix = click.style(f'[{dt}] SNCO {label}:', fg=fmt['fg'])
    return f'{prefix} {msg}'


class LogFormatter(logging.Formatter):

    def format(self, record):
        if not record.exc_info:
            lvl = record.levelname
            msg = record.getMessage()
            msg = '\n'.join(format_log_msg(lvl, line) for line in msg.splitlines())
            return msg
        return logging.Formatter.format(self, record)


class ClickLogHandler(logging.Handler):

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname.upper()
            click.echo(msg, err=True)
        except Exception:
            self.handleError(record)


def click_logger(ctx, params, value):
    log = logging.getLogger('snco')
    log.setLevel(value)
    log_handler = ClickLogHandler()
    log_handler.formatter = LogFormatter()
    log.handlers = [log_handler]
    log.propagate = False
    return log


def progress_bar(iterable, label=None, **kwargs):
    if label is not None:
        label = format_log_msg('INFO', label)
    return click.progressbar(iterable, label=label, **kwargs)
