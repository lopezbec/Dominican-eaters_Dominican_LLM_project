"""Optional terminal interface for discovering and running CLI workflows."""

from __future__ import annotations


class TUIUnavailableError(RuntimeError):
    """Raised when the optional terminal-interface dependency is unavailable."""


def run_tui() -> None:
    """Start the terminal interface without importing Textual during normal CLI use."""

    try:
        from .app import DominicanEatersApp
    except ModuleNotFoundError as error:
        if error.name == "textual" or (error.name and error.name.startswith("textual.")):
            raise TUIUnavailableError(
                "The terminal interface is not installed. Install it with "
                "`python -m pip install 'dominican-eaters[tui]'`."
            ) from error
        raise
    DominicanEatersApp().run()


__all__ = ["TUIUnavailableError", "run_tui"]
