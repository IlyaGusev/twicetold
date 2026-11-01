from jinja2 import Environment, FileSystemLoader, StrictUndefined

from settings import settings

_env = Environment(loader=FileSystemLoader(settings.PROMPTS_DIR), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def render(name, **values):
    return _env.get_template(f"{name}.j2").render(**values)
