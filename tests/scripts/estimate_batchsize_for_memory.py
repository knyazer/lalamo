import traceback as tb

from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models.language_model import LanguageModelConfig
from lalamo.speculator.estimator import estimate_batchsize_from_memory
from pathlib import Path

MODEL_LIST = ["models/Gemma-3-1B-Instruct-8bit"]

MEMORY_GIB = [4]
KINDS = ["hybrid", "cpu"]

HEADERS = [
    "model",
    "kind",
    "target_mem",
    "batch_size",
]


def _load_model(model_ref: str):
    path = Path(model_ref)
    if path.exists():
        return LanguageModelConfig.load_model(path)
    raise ValueError(f"Unknown model reference: {model_ref!r}")


def _render_table(rows: list[list[str]]) -> str:
    widths = [len(h) for h in HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(HEADERS)))
    body_lines = [" | ".join(row[i].ljust(widths[i]) for i in range(len(HEADERS))) for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def _format_mem_gib(mem_gib: int) -> str:
    return f"{mem_gib} GiB"


def main() -> None:
    rows: list[list[str]] = []
    for model_ref in MODEL_LIST:
        model = _load_model(model_ref)
        for mem_gib in MEMORY_GIB:
            for kind in KINDS:
                target_mem = mem_gib * 1024**3
                try:
                    batch_size = estimate_batchsize_from_memory(
                        model,
                        max_input_length=128,
                        max_output_length=32,
                        num_logits_per_token=8,
                        target_mem=target_mem,
                        kind=kind,
                    )
                    batch_size_str = str(batch_size)
                except Exception:
                    batch_size_str = f"error: {tb.format_exc()}"

                rows.append(
                    [
                        model_ref,
                        kind,
                        _format_mem_gib(mem_gib),
                        batch_size_str,
                    ]
                )
                print(_render_table(rows))
                print()


if __name__ == "__main__":
    main()
