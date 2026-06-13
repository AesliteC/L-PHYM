from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json


def _parse_label_reason(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if ":" in value:
            label, reason = value.split(":", 1)
        else:
            label, reason = value, "manual_quality_override"
        label = label.strip()
        reason = reason.strip() or "manual_quality_override"
        if not label:
            raise ValueError(f"empty override label in {value!r}")
        parsed[label] = reason
    return parsed


def apply_quality_overrides(
    payload: dict[str, object],
    *,
    include: dict[str, str] | None = None,
    exclude: dict[str, str] | None = None,
) -> dict[str, object]:
    include = include or {}
    exclude = exclude or {}
    if set(include) & set(exclude):
        overlap = ", ".join(sorted(set(include) & set(exclude)))
        raise ValueError(f"labels cannot be both included and excluded: {overlap}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for source_row in payload.get("rows", []):  # type: ignore[arg-type]
        row = dict(source_row)
        label = str(row.get("label", ""))
        seen.add(label)
        overrides: list[dict[str, object]] = list(row.get("manual_overrides", []))  # type: ignore[arg-type]
        if label in include:
            row["accepted"] = True
            row["reject_reasons"] = []
            overrides.append({"action": "include", "reason": include[label]})
        if label in exclude:
            row["accepted"] = False
            reasons = [str(reason) for reason in row.get("reject_reasons", [])]  # type: ignore[arg-type]
            manual_reason = f"manual_exclude:{exclude[label]}"
            if manual_reason not in reasons:
                reasons.append(manual_reason)
            row["reject_reasons"] = reasons
            overrides.append({"action": "exclude", "reason": exclude[label]})
        if overrides:
            row["manual_overrides"] = overrides
        rows.append(row)

    missing = sorted((set(include) | set(exclude)) - seen)
    if missing:
        raise ValueError(f"override labels not found in quality summary: {', '.join(missing)}")

    accepted = [row for row in rows if row.get("accepted")]
    rejected = [row for row in rows if not row.get("accepted")]
    out = dict(payload)
    out["manual_override_summary"] = {
        "include": include,
        "exclude": exclude,
    }
    out["counts"] = {
        "total": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
    }
    out["accepted_paths"] = [str(row["path"]) for row in accepted]
    out["rejected_labels"] = [str(row.get("label", "")) for row in rejected]
    out["rows"] = rows
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-summary", required=True)
    parser.add_argument("--include", action="append", default=[], help="Label or 'label:reason' to force accept.")
    parser.add_argument("--exclude", action="append", default=[], help="Label or 'label:reason' to force reject.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--quiet", action="store_true", help="Print only override counts and output path.")
    args = parser.parse_args(argv)

    source = Path(args.quality_summary)
    payload = json.loads(source.read_text(encoding="utf-8"))
    output_payload = apply_quality_overrides(
        payload,
        include=_parse_label_reason(args.include),
        exclude=_parse_label_reason(args.exclude),
    )
    output_payload["source_summary"] = str(source)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {
        "output_json": str(output),
        "counts": output_payload["counts"],
        "manual_override_summary": output_payload["manual_override_summary"],
    }
    print(json.dumps(compact if args.quiet else output_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
