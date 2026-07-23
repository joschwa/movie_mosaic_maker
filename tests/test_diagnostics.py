import logging

from movie_mosaic_maker.diagnostics import Diagnostics, check_leftover_fraction, check_passes


def test_check_leftover_fraction_below_threshold_returns_none() -> None:
    assert check_leftover_fraction(0.05) is None
    assert check_leftover_fraction(0.15) is None  # exactly at threshold: not a warning


def test_check_leftover_fraction_above_threshold_returns_message() -> None:
    message = check_leftover_fraction(0.32)
    assert message is not None
    assert "32.0%" in message


def test_check_leftover_fraction_respects_custom_threshold() -> None:
    assert check_leftover_fraction(0.05, threshold=0.01) is not None
    assert check_leftover_fraction(0.05, threshold=0.5) is None


def test_check_passes_single_pass_returns_none() -> None:
    assert check_passes(1) is None
    assert check_passes(0) is None


def test_check_passes_multi_pass_returns_message() -> None:
    message = check_passes(3)
    assert message is not None
    assert "3 passes" in message


def test_diagnostics_captures_warnings_from_package_loggers() -> None:
    diagnostics = Diagnostics()
    diagnostics.attach()
    try:
        logging.getLogger("movie_mosaic_maker.images").warning("could not load image bad.png")
    finally:
        diagnostics.detach()

    assert any("bad.png" in w for w in diagnostics.warnings)
    assert diagnostics.has_warnings()


def test_diagnostics_does_not_capture_before_attach_or_after_detach() -> None:
    diagnostics = Diagnostics()
    logging.getLogger("movie_mosaic_maker.images").warning("before attach")

    diagnostics.attach()
    diagnostics.detach()
    logging.getLogger("movie_mosaic_maker.images").warning("after detach")

    assert diagnostics.warnings == []
    assert not diagnostics.has_warnings()


def test_diagnostics_context_manager_attaches_and_detaches() -> None:
    with Diagnostics() as diagnostics:
        logging.getLogger("movie_mosaic_maker.video").warning("seek drift happened")
    assert any("seek drift" in w for w in diagnostics.warnings)

    # after the context exits, this instance's handler must be detached
    logging.getLogger("movie_mosaic_maker.video").warning("should not be captured")
    assert not any("should not be captured" in w for w in diagnostics.warnings)


def test_diagnostics_instances_do_not_leak_into_each_other() -> None:
    d1 = Diagnostics()
    d2 = Diagnostics()
    d1.attach()
    try:
        logging.getLogger("movie_mosaic_maker.images").warning("only for d1")
    finally:
        d1.detach()

    assert any("only for d1" in w for w in d1.warnings)
    assert d2.warnings == []


def test_diagnostics_note_and_threshold_helpers_combine_in_report() -> None:
    diagnostics = Diagnostics()
    diagnostics.note("manual note")
    diagnostics.note_leftover_fraction(0.5)
    diagnostics.note_passes(4)
    diagnostics.note_leftover_fraction(0.05)  # below threshold, should add nothing
    diagnostics.note_passes(1)  # single pass, should add nothing

    assert diagnostics.has_warnings()
    assert len(diagnostics.warnings) == 3
    report = diagnostics.report()
    assert "3 warning(s):" in report
    assert "manual note" in report
    assert "50.0%" in report
    assert "4 passes" in report


def test_diagnostics_report_with_no_warnings() -> None:
    diagnostics = Diagnostics()
    assert diagnostics.report() == "No warnings."
    assert not diagnostics.has_warnings()
