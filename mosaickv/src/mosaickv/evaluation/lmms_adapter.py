"""lmms-eval 0.7.2 adapter for instrumented local MosaicKV models.

The module imports lmms-eval lazily so the CPU-only core package and synthetic
CI task do not require its large dependency graph.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from numbers import Real
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.model import (
    EvaluationRequest,
    LocalEvaluationModel,
    ModelGeneration,
)
from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.storage import JsonlResultStore, write_parquet_aggregate
from mosaickv.evaluation.tasks import (
    TaskRegistry,
    TaskSpec,
    default_task_registry,
    deterministic_sample_ids,
)
from mosaickv.types import JsonObject


class LmmsEvalUnavailable(RuntimeError):
    """Raised when an lmms-eval runtime is required but unavailable."""


_LMMS_EVAL_VERSION = "0.7.2"


@dataclass(frozen=True, slots=True)
class CapturedGeneration:
    """MosaicKV side-channel observation captured during lmms generation."""

    sample_id: str
    lmms_task: str
    generation: ModelGeneration | None
    error: str | None


@dataclass(frozen=True, slots=True)
class PreparedLmmsSample:
    """Identity and reference retained before lmms-eval begins execution."""

    sample_id: str
    public_task: str
    lmms_task: str
    reference: str


def _require_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise LmmsEvalUnavailable(
            "lmms-eval 0.7.2 is required; use the locked HF evaluation environment"
        ) from error


def _require_lmms_version() -> None:
    try:
        installed = version("lmms-eval")
    except PackageNotFoundError as error:
        raise LmmsEvalUnavailable(
            "lmms-eval 0.7.2 is required; use the locked HF evaluation environment"
        ) from error
    if installed != _LMMS_EVAL_VERSION:
        raise LmmsEvalUnavailable(
            f"lmms-eval {_LMMS_EVAL_VERSION} is required, but {installed} is installed"
        )


@contextmanager
def _pinned_dataset_revision(dataset_id: str, revision: str) -> Iterator[None]:
    """Inject one immutable Hub revision into lmms-eval's dataset load.

    lmms-eval 0.7.2 constructs and downloads a task inside
    ``get_task_dict``.  Its MMStar YAML does not pin ``dataset_kwargs``.  The
    scoped patch targets only the audited dataset ID and leaves every other
    ``datasets.load_dataset`` call unchanged.
    """

    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise ValueError("dataset revision must be a 40-64 character lowercase commit SHA")
    datasets_module = _require_module("datasets")
    original = datasets_module.load_dataset
    observed = 0

    def pinned(path: object, *args: object, **kwargs: object) -> object:
        nonlocal observed
        if str(path) == dataset_id:
            configured = kwargs.get("revision")
            if configured is not None and configured != revision:
                raise ValueError(
                    f"lmms task requested dataset revision {configured!r}, expected {revision!r}"
                )
            kwargs["revision"] = revision
            if kwargs.get("token") is True and not os.environ.get("HF_TOKEN"):
                # The audited dataset is public.  lmms-eval's legacy template
                # uses token=True, which newer Hub clients interpret as
                # "require a locally persisted login".  Never create/read a
                # token file; use HF_TOKEN when present and anonymous access
                # otherwise.
                kwargs["token"] = False
            observed += 1
        return original(path, *args, **kwargs)

    with patch.object(datasets_module, "load_dataset", pinned):
        yield
    if observed == 0:
        raise RuntimeError(f"lmms task did not load its declared dataset {dataset_id!r}")


@contextmanager
def _lmms_template_compatibility() -> Iterator[None]:
    """Redirect known missing lmms-eval 0.7.2 wheel templates to pinned copies."""

    lmms_utils = _require_module("lmms_eval.utils")
    original = lmms_utils.load_yaml_config
    compatibility_root = files("mosaickv.evaluation.lmms_compat")

    def load_yaml_config(*args: object, **kwargs: object) -> object:
        yaml_path = kwargs.get("yaml_path")
        positional = list(args)
        if yaml_path is None and positional:
            yaml_path = positional[0]
        if yaml_path is not None:
            requested = Path(str(yaml_path))
            if not requested.is_file() and requested.name == "_default_template_yaml":
                replacement = compatibility_root.joinpath(requested.parent.name, requested.name)
                if replacement.is_file():
                    if positional:
                        positional[0] = str(replacement)
                    else:
                        kwargs["yaml_path"] = str(replacement)
                    kwargs.setdefault("yaml_dir", str(requested.parent))
        return original(*positional, **kwargs)

    with patch.object(lmms_utils, "load_yaml_config", load_yaml_config):
        yield


def _flatten_visuals(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return tuple(value[0])
        return tuple(value)
    return (value,)


def _media_kind(value: object) -> MediaKind:
    if isinstance(value, (str, Path)):
        suffix = Path(str(value)).suffix.casefold()
        if suffix in {".avi", ".mkv", ".mov", ".mp4", ".webm"}:
            return MediaKind.VIDEO
    return MediaKind.IMAGE


def _sample_id(task_name: str, doc_id: object, doc: object) -> str:
    if isinstance(doc, dict):
        preserved = doc.get("_mosaickv_sample_id")
        if isinstance(preserved, str) and preserved:
            return preserved
    return f"{task_name}:{doc_id}"


class LmmsRequestBridge:
    """Translate lmms ``Instance.args`` into the local model protocol."""

    def __init__(self, model: LocalEvaluationModel, run_id: str) -> None:
        self.model = model
        self.run_id = run_id
        self._captures: dict[str, CapturedGeneration] = {}

    @property
    def captures(self) -> tuple[CapturedGeneration, ...]:
        return tuple(self._captures[key] for key in sorted(self._captures))

    def generate_until(self, requests: list[object], task_dict: Mapping[str, object]) -> list[str]:
        """Generate responses in input order and retain failures as captures."""

        responses: list[str] = []
        for request in requests:
            args = getattr(request, "args", None)
            if not isinstance(args, tuple) or len(args) < 6:
                raise ValueError(
                    "lmms generate_until request must expose six arguments: "
                    "context, kwargs, doc_to_visual, doc_id, task, split"
                )
            context, generation_kwargs, doc_to_visual, doc_id, task_name, split = args[:6]
            if not isinstance(context, str) or not isinstance(generation_kwargs, dict):
                raise TypeError("lmms context and generation kwargs have unexpected types")
            if not isinstance(task_name, str) or not isinstance(split, str):
                raise TypeError("lmms task and split must be strings")
            dataset = task_dict[task_name]
            doc = dataset[split][doc_id]  # type: ignore[index]
            sample_id = _sample_id(task_name, doc_id, doc)
            if sample_id in self._captures:
                raise ValueError(
                    f"duplicate lmms generation for sample {sample_id!r}; repeats must be 1"
                )
            try:
                visuals = _flatten_visuals(doc_to_visual(doc))
                media = tuple(MediaItem(_media_kind(visual), visual) for visual in visuals)
                has_video = any(item.kind == MediaKind.VIDEO for item in media)
                if has_video and not self.model.supports_video:
                    raise ValueError(f"model {self.model.model_id!r} does not support video")
                generation = self.model.generate(
                    EvaluationRequest(
                        run_id=self.run_id,
                        sample_id=sample_id,
                        task=task_name,
                        messages=build_multimodal_messages(context, media),
                        generation_kwargs=cast("dict[str, object]", generation_kwargs.copy()),
                    )
                )
                capture = CapturedGeneration(sample_id, task_name, generation, None)
                response = generation.answer
            except Exception as error:  # lmms must receive one response per request.
                capture = CapturedGeneration(
                    sample_id,
                    task_name,
                    None,
                    f"{type(error).__name__}: {error}",
                )
                response = ""
            self._captures[sample_id] = capture
            responses.append(response)
        return responses


def create_lmms_model_adapter(model: LocalEvaluationModel, *, run_id: str) -> object:
    """Create an instantiated lmms-eval model backed by ``model``.

    The returned object is passed directly to ``lmms_eval.evaluator.simple_evaluate``;
    no global model registry mutation is required.
    """

    _require_lmms_version()
    model_api = _require_module("lmms_eval.api.model")
    base = model_api.lmms
    bridge = LmmsRequestBridge(model, run_id)

    class LmmsMosaicKVAdapter(base):  # type: ignore[misc, valid-type]
        is_simple = True

        def __init__(self) -> None:
            super().__init__()
            self.mosaickv_bridge = bridge

        def generate_until(self, requests: list[object]) -> list[str]:
            return self.mosaickv_bridge.generate_until(requests, self.task_dict)

        def loglikelihood(self, requests: list[object]) -> list[tuple[float, bool]]:
            raise NotImplementedError("MosaicKV lmms adapter supports generate_until tasks only")

        def generate_until_multi_round(self, requests: list[object]) -> list[str]:
            raise NotImplementedError("multi-round generation is not supported")

    return LmmsMosaicKVAdapter()


def _source_ids(dataset: Any, public_task: str) -> list[str]:
    columns = set(dataset.column_names)
    id_field = next(
        (field for field in ("question_id", "questionId", "index", "id") if field in columns),
        None,
    )
    raw_ids = dataset[id_field] if id_field is not None else range(len(dataset))
    source_ids: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_ids):
        candidate = str(value) if isinstance(value, (str, int)) else str(index)
        sample_id = f"{public_task}:{candidate}"
        if sample_id in seen:
            sample_id = f"{sample_id}:row-{index}"
        seen.add(sample_id)
        source_ids.append(sample_id)
    return source_ids


def _canonical_reference(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _prepare_task(
    task: Any,
    *,
    public_task: str,
    lmms_task: str,
    seed: int,
    subset_size: int,
    excluded: frozenset[str],
) -> tuple[Any | None, tuple[PreparedLmmsSample, ...], tuple[str, ...]]:
    split = task.config.test_split or task.config.validation_split
    if not isinstance(split, str):
        raise ValueError(f"lmms task {lmms_task!r} has no test or validation split")
    dataset = task.dataset[split]
    source_ids = _source_ids(dataset, public_task)
    selected_ids = deterministic_sample_ids(source_ids, seed=seed, subset_size=subset_size)
    index_by_id = {sample_id: index for index, sample_id in enumerate(source_ids)}
    pending_ids = tuple(sample_id for sample_id in selected_ids if sample_id not in excluded)
    if not pending_ids:
        return None, (), selected_ids
    indices = [index_by_id[sample_id] for sample_id in pending_ids]
    subset = dataset.select(indices).add_column("_mosaickv_sample_id", list(pending_ids))
    task.dataset[split] = subset
    if hasattr(task, "dataset_no_image") and split in task.dataset_no_image:
        task.dataset_no_image[split] = (
            task.dataset_no_image[split]
            .select(indices)
            .add_column("_mosaickv_sample_id", list(pending_ids))
        )
    task.task_docs = subset
    prepared = tuple(
        PreparedLmmsSample(
            sample_id=sample_id,
            public_task=public_task,
            lmms_task=lmms_task,
            reference=_canonical_reference(task.doc_to_target(subset[position])),
        )
        for position, sample_id in enumerate(pending_ids)
    )
    return task, prepared, selected_ids


def prepare_seeded_lmms_tasks(
    specs: tuple[TaskSpec, ...],
    *,
    seed: int,
    subset_size: int,
    excluded: frozenset[str] = frozenset(),
    model_name: str | None = None,
    dataset_revisions: Mapping[str, str] | None = None,
) -> tuple[tuple[object, ...], tuple[PreparedLmmsSample, ...], tuple[str, ...], object]:
    """Load lmms task objects and replace their eval splits with seeded subsets."""

    _require_lmms_version()
    if subset_size < 1:
        raise ValueError("subset_size must be >= 1")
    tasks_module = _require_module("lmms_eval.tasks")
    tasks_root = Path(tasks_module.__file__).resolve().parent
    include_paths: set[str] = set()
    for spec in specs:
        task_name = spec.development_lmms_task
        if task_name is None:
            raise ValueError(f"task {spec.name!r} is not an lmms-eval task")
        task_directory = tasks_root / task_name.split("_", maxsplit=1)[0]
        if not task_directory.is_dir():
            raise RuntimeError(
                f"lmms-eval task directory is unavailable for {task_name!r}: {task_directory}"
            )
        include_paths.add(str(task_directory))
    # Index only requested task families.  Besides being substantially faster,
    # this isolates evaluation from packaging defects in unrelated task YAMLs
    # (lmms-eval 0.7.2's CV-Bench wheel omits an included template file).
    with _lmms_template_compatibility():
        manager = tasks_module.TaskManager(
            verbosity="ERROR",
            include_path=sorted(include_paths),
            include_defaults=False,
            model_name=model_name,
        )
    selected_tasks: list[object] = []
    prepared: list[PreparedLmmsSample] = []
    selected_ids: list[str] = []
    for spec in specs:
        task_name = spec.development_lmms_task
        if task_name is None:
            raise ValueError(f"task {spec.name!r} is not an lmms-eval task")
        revision = None if dataset_revisions is None else dataset_revisions.get(spec.dataset_id)
        if dataset_revisions is not None and revision is None:
            raise ValueError(f"no immutable revision supplied for dataset {spec.dataset_id!r}")
        if revision is None:
            with _lmms_template_compatibility():
                loaded = tasks_module.get_task_dict(
                    [task_name], task_manager=manager, task_type="simple"
                )
        else:
            with (
                _lmms_template_compatibility(),
                _pinned_dataset_revision(spec.dataset_id, revision),
            ):
                loaded = tasks_module.get_task_dict(
                    [task_name], task_manager=manager, task_type="simple"
                )
        task = loaded[task_name]
        if isinstance(task, tuple):
            task = task[-1]
        subset_task, task_samples, task_selected_ids = _prepare_task(
            task,
            public_task=spec.name,
            lmms_task=task_name,
            seed=seed,
            subset_size=subset_size,
            excluded=excluded,
        )
        if subset_task is not None:
            selected_tasks.append(subset_task)
            prepared.extend(task_samples)
        selected_ids.extend(task_selected_ids)
    return tuple(selected_tasks), tuple(prepared), tuple(selected_ids), manager


def _extract_answer(sample: dict[str, object]) -> str:
    responses = sample.get("filtered_resps")
    value: object = responses
    while isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, str):
        raise ValueError("lmms sample does not contain a string filtered response")
    return value


def _extract_score(sample: dict[str, object], metric: str) -> float:
    value = sample.get(metric)
    if isinstance(value, dict):
        value = value.get("score")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"lmms metric {metric!r} is not a numeric per-sample score")
    return float(value)


def _extract_preserved_id(sample: dict[str, object], lmms_task: str) -> str:
    doc = sample.get("doc")
    if isinstance(doc, dict):
        value = doc.get("_mosaickv_sample_id")
        if isinstance(value, str):
            return value
    return f"{lmms_task}:{sample.get('doc_id')}"


def run_lmms_development_evaluation(
    *,
    run_id: str,
    task_names: tuple[str, ...],
    model: LocalEvaluationModel,
    raw_output: str | Path,
    manifest_path: str,
    seed: int,
    subset_size: int,
    parquet_output: str | Path | None = None,
    registry: TaskRegistry | None = None,
    dataset_revision: str | None = None,
) -> JsonObject:
    """Run deterministic development subsets through lmms-eval scoring."""

    if not task_names:
        raise ValueError("at least one task name is required")
    if len(set(task_names)) != len(task_names):
        raise ValueError("task names must be unique")
    selected_registry = registry or default_task_registry()
    specs = tuple(selected_registry.resolve(name) for name in task_names)
    for spec in specs:
        if spec.requires_video and not model.supports_video:
            raise ValueError(f"model {model.model_id!r} does not support task {spec.name!r} video")
    store = JsonlResultStore(raw_output)
    prior_ids = store.completed_sample_ids(run_id)
    task_objects, prepared, selected_ids, task_manager = prepare_seeded_lmms_tasks(
        specs,
        seed=seed,
        subset_size=subset_size,
        excluded=prior_ids,
        model_name=model.model_id,
        dataset_revisions=(
            None
            if dataset_revision is None
            else {spec.dataset_id: dataset_revision for spec in specs}
        ),
    )
    spec_by_lmms = {cast("str", spec.development_lmms_task): spec for spec in specs}
    expected = {sample.sample_id: sample for sample in prepared}
    adapter = create_lmms_model_adapter(model, run_id=run_id)
    bridge = cast("LmmsRequestBridge", adapter.mosaickv_bridge)  # type: ignore[attr-defined]
    lmms_error: str | None = None
    evaluated: dict[str, object] = {}
    if task_objects:
        try:
            evaluator = _require_module("lmms_eval.evaluator")
            result = evaluator.simple_evaluate(
                model=adapter,
                tasks=list(task_objects),
                task_manager=task_manager,
                num_fewshot=0,
                batch_size=1,
                limit=None,
                bootstrap_iters=0,
                log_samples=True,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                fewshot_random_seed=seed,
                repeats=1,
            )
            if not isinstance(result, dict) or not isinstance(result.get("samples"), dict):
                raise RuntimeError("lmms-eval returned no per-sample results")
            evaluated = result["samples"]
        except Exception as error:  # All pending samples are materialized below.
            lmms_error = f"{type(error).__name__}: {error}"

    captures = {capture.sample_id: capture for capture in bridge.captures}
    written: set[str] = set()
    if lmms_error is None:
        for lmms_task, rows in evaluated.items():
            if not isinstance(lmms_task, str) or not isinstance(rows, list):
                continue
            spec = spec_by_lmms[lmms_task]
            for raw_sample in rows:
                if not isinstance(raw_sample, dict):
                    continue
                sample = cast("dict[str, object]", raw_sample)
                sample_id = _extract_preserved_id(sample, lmms_task)
                prepared_sample = expected.get(sample_id)
                if prepared_sample is None:
                    continue
                capture = captures.get(sample_id)
                try:
                    if capture is None:
                        raise RuntimeError("lmms sample has no MosaicKV telemetry capture")
                    if capture.error is not None or capture.generation is None:
                        raise RuntimeError(capture.error or "generation failed without an error")
                    answer = _extract_answer(sample)
                    row = EvaluationResult.from_generation(
                        run_id=run_id,
                        sample_id=sample_id,
                        task=spec.name,
                        model=model.model_id,
                        backend=model.backend,
                        method=capture.generation.effective_method or model.method,
                        retention_ratio=model.retention_ratio,
                        answer=answer,
                        reference=prepared_sample.reference,
                        task_score=_extract_score(sample, spec.sample_metric),
                        metrics=capture.generation.metrics,
                        manifest_path=manifest_path,
                    )
                except Exception as error:  # Conversion failure becomes a recorded row.
                    row = EvaluationResult.failed(
                        run_id=run_id,
                        sample_id=sample_id,
                        task=spec.name,
                        model=model.model_id,
                        backend=model.backend,
                        method=(
                            model.method
                            if capture is None or capture.generation is None
                            else capture.generation.effective_method or model.method
                        ),
                        retention_ratio=model.retention_ratio,
                        answer=(
                            None
                            if capture is None or capture.generation is None
                            else capture.generation.answer
                        ),
                        reference=prepared_sample.reference,
                        error=f"{type(error).__name__}: {error}",
                        manifest_path=manifest_path,
                        metrics=(
                            None
                            if capture is None or capture.generation is None
                            else capture.generation.metrics
                        ),
                    )
                store.append(row)
                written.add(sample_id)

    for sample_id, prepared_sample in expected.items():
        if sample_id in written:
            continue
        capture = captures.get(sample_id)
        pending_error = lmms_error or (capture.error if capture is not None else None)
        store.append(
            EvaluationResult.failed(
                run_id=run_id,
                sample_id=sample_id,
                task=prepared_sample.public_task,
                model=model.model_id,
                backend=model.backend,
                method=(
                    model.method
                    if capture is None or capture.generation is None
                    else capture.generation.effective_method or model.method
                ),
                retention_ratio=model.retention_ratio,
                answer=(
                    None
                    if capture is None or capture.generation is None
                    else capture.generation.answer
                ),
                reference=prepared_sample.reference,
                error=pending_error or "lmms-eval did not return this selected sample",
                manifest_path=manifest_path,
                metrics=(
                    None
                    if capture is None or capture.generation is None
                    else capture.generation.metrics
                ),
            )
        )

    rows = store.results(run_id=run_id)
    parquet_path = None
    if parquet_output is not None:
        parquet_path = str(write_parquet_aggregate(rows, parquet_output).resolve())
    return {
        "run_id": run_id,
        "tasks": list(task_names),
        "selected_samples": len(selected_ids),
        "resumed_samples": len(set(selected_ids) & prior_ids),
        "raw_output": str(Path(raw_output).resolve()),
        "parquet_output": parquet_path,
        "lmms_eval_version": _LMMS_EVAL_VERSION,
        "dataset_revisions": {
            spec.dataset_id: dataset_revision for spec in specs if dataset_revision is not None
        },
        "lmms_error": lmms_error,
        "nondeterministic_scoring_tasks": [
            spec.name for spec in specs if not spec.scoring_deterministic
        ],
    }


__all__ = [
    "CapturedGeneration",
    "LmmsEvalUnavailable",
    "LmmsRequestBridge",
    "PreparedLmmsSample",
    "create_lmms_model_adapter",
    "prepare_seeded_lmms_tasks",
    "run_lmms_development_evaluation",
]
