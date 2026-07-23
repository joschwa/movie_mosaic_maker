from __future__ import annotations

import logging

LEFTOVER_FRACTION_WARNING_THRESHOLD = 0.15
PACKAGE_LOGGER_NAME = "movie_mosaic_maker"


def check_leftover_fraction(leftover_fraction: float, threshold: float = LEFTOVER_FRACTION_WARNING_THRESHOLD) -> str | None:
    if leftover_fraction <= threshold:
        return None
    return (
        f"{leftover_fraction:.1%} of the target canvas is wasted by the chosen grid shape -- "
        "the sample pool's aspect ratio may poorly match the target image's aspect ratio"
    )


def check_passes(passes: int) -> str | None:
    if passes <= 1:
        return None
    return (
        f"assignment needed {passes} passes over the candidate pool -- --max-reuse is low "
        "relative to grid size vs. pool size, expect visible repetition"
    )


class _WarningCollectorHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class Diagnostics:
    """Collects warnings from across a single pipeline run -- both incidental
    ones logged deep in other modules (skipped files, video seek drift) and
    explicit threshold checks on values only known at the orchestration layer
    (grid leftover fraction, assignment pass count) -- so they can be printed
    as one consolidated summary at the end instead of scattered mid-run lines.
    """

    def __init__(self) -> None:
        self._handler = _WarningCollectorHandler()
        self._notes: list[str] = []

    def attach(self, logger_name: str = PACKAGE_LOGGER_NAME) -> None:
        logging.getLogger(logger_name).addHandler(self._handler)

    def detach(self, logger_name: str = PACKAGE_LOGGER_NAME) -> None:
        logging.getLogger(logger_name).removeHandler(self._handler)

    def __enter__(self) -> Diagnostics:
        self.attach()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.detach()

    def note(self, message: str) -> None:
        self._notes.append(message)

    def note_leftover_fraction(self, leftover_fraction: float, threshold: float = LEFTOVER_FRACTION_WARNING_THRESHOLD) -> None:
        message = check_leftover_fraction(leftover_fraction, threshold)
        if message is not None:
            self.note(message)

    def note_passes(self, passes: int) -> None:
        message = check_passes(passes)
        if message is not None:
            self.note(message)

    @property
    def warnings(self) -> list[str]:
        return [record.getMessage() for record in self._handler.records] + list(self._notes)

    def has_warnings(self) -> bool:
        return bool(self._handler.records) or bool(self._notes)

    def report(self) -> str:
        messages = self.warnings
        if not messages:
            return "No warnings."
        lines = [f"{len(messages)} warning(s):"]
        lines.extend(f"  - {message}" for message in messages)
        return "\n".join(lines)
