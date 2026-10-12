"""Concurrent job runner that grades experiment setups and streams results to disk."""

from __future__ import annotations

import logging
import threading

from langchain_core.runnables import RunnableLambda

from abaes_eval_fwk.core.grader.grader import Grader
from abaes_eval_fwk.core.llm.llm_loader import LLMLoader
from abaes_eval_fwk.core.retriever.retrieval_loader import RetrievalLoader
from abaes_eval_fwk.experiment.result.results_saver import ResultsWriter
from abaes_eval_fwk.experiment.setup.reuse_loader import (
    build_reuse_by_setup,
    build_review_cache,
)

_flush_lock = threading.Lock()
_thread_local = threading.local()


class _BufferingHandler(logging.Handler):
    """Accumulates log records per-thread; flushes them atomically to the root stream handlers."""

    def emit(self, record: logging.LogRecord) -> None:
        buf = getattr(_thread_local, "log_buffer", None)
        if buf is not None:
            buf.append(record)
        else:
            for handler in _root_stream_handlers:
                handler.emit(record)


_root_stream_handlers: list[logging.StreamHandler] = []
_buffering_handler: _BufferingHandler | None = None


def _install_buffering_handler() -> None:
    """Replace the root's stream handlers (console, file, ...) with the buffering handler (idempotent)."""
    global _buffering_handler
    if _buffering_handler is not None:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            _root_stream_handlers.append(h)
            root.removeHandler(h)
    if not _root_stream_handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
        )
        _root_stream_handlers.append(handler)
    _buffering_handler = _BufferingHandler()
    root.addHandler(_buffering_handler)


def _flush_buffer(header: str) -> None:
    """Flush the current thread's log buffer atomically to every root stream handler."""
    buf = getattr(_thread_local, "log_buffer", [])
    if not buf:
        return
    with _flush_lock:
        for handler in _root_stream_handlers:
            handler.stream.write(f"{header}\n")
            for record in buf:
                handler.emit(record)
            handler.stream.flush()
    del _thread_local.log_buffer


logger = logging.getLogger(__name__)


def grade_dataframe(
    grader: Grader,
    max_workers: int,
    run_id: str = "",
    results_dir: str = "",
) -> tuple[str, str | None]:
    """Grade all setups attached to an experiment concurrently, streaming results (and any failures) incrementally to CSV."""

    reuse_lookup = build_reuse_by_setup(grader.experiment, grader.reuse)

    llm = LLMLoader.load(grader.experiment.llm)  # Built once, shared across jobs
    RetrievalLoader.preload_indexes(
        grader.experiment.setups, grader.experiment.retrieval
    )  # Avoid thread-race on first access

    jobs = [
        (setup, human_assessment)
        for setup in grader.experiment.setups
        for human_assessment in setup.sample
    ]
    total = len(jobs)

    _install_buffering_handler()

    with ResultsWriter(
        grader.experiment, results_dir=results_dir, run_id=run_id, reuse=grader.reuse
    ) as writer:
        review_cache = build_review_cache(grader.experiment, grader.reuse, writer)

        def _run_job(pos: int, setup, human_assessment) -> None:
            _thread_local.log_buffer = []
            essay_id = human_assessment.essay_id

            logger.info(
                "Assignment %d/%d  [id=%s  set=%s  human_score=%s]",
                pos + 1,
                total,
                essay_id,
                setup.rubric.rubric_id,
                human_assessment.human_score,
            )

            try:
                is_criterion, reuse_records, independent_records = reuse_lookup(setup, essay_id)

                record = grader.grade(
                    rubric=setup.rubric,
                    human_assessment=human_assessment,
                    llm=llm,
                    calibration_pool=setup.calibration_pool,
                    reuse_records=reuse_records,
                    independent_records=independent_records,
                    is_criterion=is_criterion,
                    review_cache=review_cache,
                )

                logger.info(record.log_summary)
                logger.debug(record.log_detail)
                writer.append(record, setup.rubric)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Assignment id=%s set=%s FAILED — skipping. %s",
                    essay_id,
                    setup.rubric.rubric_id,
                    exc,
                )
                writer.append_failure(human_assessment, setup.rubric, str(exc))
            finally:
                _flush_buffer(
                    f"\n{'─' * 60}\n"
                    f"Essay {pos + 1}/{total}  id={essay_id}  set={setup.rubric.rubric_id}  human={human_assessment.human_score}\n"
                    f"{'─' * 60}"
                )

        run_job: RunnableLambda = RunnableLambda(
            lambda job: _run_job(*job)
        )  # batch_as_completed() yields as each finishes
        for _ in run_job.batch_as_completed(
            [(pos, setup, submission) for pos, (setup, submission) in enumerate(jobs)],
            config={"max_concurrency": max_workers},
        ):
            pass
        csv_path = writer.csv_path
        failures_csv_path = writer.failures_csv_path if writer.has_failures else None

    return csv_path, failures_csv_path
