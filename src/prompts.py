"""Render a prompt template from prompts/<name>.j2. Templates keep no trailing newline, so the text is byte-stable."""

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from settings import settings

_env = Environment(loader=FileSystemLoader(settings.PROMPTS_DIR), undefined=StrictUndefined,
                   keep_trailing_newline=False, trim_blocks=True, lstrip_blocks=True)


def render(name, **values):
    return _env.get_template(f"{name}.j2").render(**values)
