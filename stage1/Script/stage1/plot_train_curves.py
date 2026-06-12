from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import csv
import json


def load_train_log(path: Path, allow_duplicate_epochs: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_epochs: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        epoch = int(row["epoch"])
        if epoch in seen_epochs and not allow_duplicate_epochs:
            raise ValueError(
                f"duplicate epoch {epoch} in {path} at line {line_no}; "
                "this usually means multiple training processes wrote the same log"
            )
        seen_epochs.add(epoch)
        rows.append(row)
    if not rows:
        raise ValueError(f"empty train log: {path}")
    return rows


def _metric(row: dict[str, object], split: str, name: str) -> float | None:
    metrics = row.get(split)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    return None if value is None else float(value)


def write_curve_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_token_accuracy",
        "val_loss",
        "val_ce_loss",
        "val_token_accuracy",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": _metric(row, "train", "loss"),
                    "train_ce_loss": _metric(row, "train", "ce_loss"),
                    "train_token_accuracy": _metric(row, "train", "token_accuracy"),
                    "val_loss": _metric(row, "val", "loss"),
                    "val_ce_loss": _metric(row, "val", "ce_loss"),
                    "val_token_accuracy": _metric(row, "val", "token_accuracy"),
                }
            )


def summarize_curve(rows: list[dict[str, object]]) -> dict[str, object]:
    best_val = None
    best_val_epoch = None
    for row in rows:
        val_loss = _metric(row, "val", "loss")
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            best_val_epoch = int(row["epoch"])
    last = rows[-1]
    return {
        "epochs": [int(row["epoch"]) for row in rows],
        "num_epochs": len(rows),
        "first_epoch": int(rows[0]["epoch"]),
        "last_epoch": int(last["epoch"]),
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val,
        "last_train_loss": _metric(last, "train", "loss"),
        "last_train_ce_loss": _metric(last, "train", "ce_loss"),
        "last_train_token_accuracy": _metric(last, "train", "token_accuracy"),
        "last_val_loss": _metric(last, "val", "loss"),
        "last_val_ce_loss": _metric(last, "val", "ce_loss"),
        "last_val_token_accuracy": _metric(last, "val", "token_accuracy"),
    }


def plot_curves(rows: list[dict[str, object]], output_png: Path, output_pdf: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [_metric(row, "train", "loss") for row in rows]
    val_loss = [_metric(row, "val", "loss") for row in rows]
    train_acc = [_metric(row, "train", "token_accuracy") for row in rows]
    val_acc = [_metric(row, "val", "token_accuracy") for row in rows]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epochs, train_loss, marker="o", label="train loss")
    if any(value is not None for value in val_loss):
        axes[0].plot(epochs, val_loss, marker="o", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train token acc")
    if any(value is not None for value in val_acc):
        axes[1].plot(epochs, val_acc, marker="o", label="val token acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, max(0.5, min(1.0, max([value or 0.0 for value in train_acc + val_acc]) * 1.15)))
    axes[1].set_title("RVQ Token Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-duplicate-epochs", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    rows = load_train_log(Path(args.train_log), allow_duplicate_epochs=args.allow_duplicate_epochs)
    write_curve_csv(rows, output_dir / "loss_accuracy_curve_data.csv")
    summary = summarize_curve(rows)
    (output_dir / "curve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_curves(
        rows,
        output_png=output_dir / "loss_accuracy_curve.png",
        output_pdf=output_dir / "loss_accuracy_curve.pdf",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
