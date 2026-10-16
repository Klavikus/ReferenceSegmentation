from dataclasses import dataclass
import os


FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


@dataclass(frozen=True)
class AppRuntimeConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    debug: bool = False
    share: bool = False


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        return default
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value!r}")


def _read_port_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip()
    if not normalized_value:
        return default

    try:
        port = int(normalized_value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {name}: {raw_value!r}") from exc

    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid TCP port for {name}: {raw_value!r}")

    return port


def load_runtime_config() -> AppRuntimeConfig:
    return AppRuntimeConfig(
        host=os.getenv("APP_HOST", DEFAULT_HOST),
        port=_read_port_env("APP_PORT", DEFAULT_PORT),
        debug=_read_bool_env("APP_DEBUG", False),
        share=_read_bool_env("APP_SHARE", False),
    )
