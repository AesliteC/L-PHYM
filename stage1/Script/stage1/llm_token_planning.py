from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import argparse
import json
import math
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_PAD_INDEX = 513
DEFAULT_VOCAB_SIZE = 512
DEFAULT_RVQ_DEPTH = 4


@dataclass
class MotionTokenExample:
    example_id: str
    caption: str
    indices: list[list[int]]
    source: str
    sequence_id: str | None = None
    window_range: list[int] | None = None
    sample_ids: list[str] | None = None

    @property
    def depth0(self) -> list[int]:
        return [int(token[0]) for token in self.indices if token]


def text_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def split_text_segments(text: str, joiner: str = " then ") -> list[str]:
    parts = [part.strip(" .,\n\t") for part in text.split(joiner)]
    return [part for part in parts if part]


def _tensor_at(cache: dict[str, object], key: str, idx: int):
    value = cache.get(key)
    if value is None:
        return None
    return value[idx]


def _valid_indices_from_cache_item(
    indices: torch.Tensor,
    target_mask: torch.Tensor | None,
    max_tokens: int,
    pad_index: int = DEFAULT_PAD_INDEX,
) -> list[list[int]]:
    if indices.ndim != 2:
        raise ValueError(f"expected one cache sample with shape (T,D), got {tuple(indices.shape)}")
    time_mask = indices[:, 0] != pad_index
    if target_mask is not None:
        if target_mask.ndim != 1 or target_mask.shape[0] != indices.shape[0]:
            raise ValueError(
                f"target_mask shape {tuple(target_mask.shape)} does not match sample shape {tuple(indices.shape)}"
            )
        time_mask = time_mask & target_mask.to(dtype=torch.bool)
    valid = indices[time_mask]
    valid = valid[:max_tokens]
    return [[int(x) for x in row.tolist()] for row in valid]


def export_example_bank_from_cache(
    cache_path: Path,
    output_path: Path,
    *,
    max_examples: int = 1600,
    max_tokens_per_example: int = 50,
    min_tokens_per_example: int = 3,
    pad_index: int = DEFAULT_PAD_INDEX,
) -> dict[str, object]:
    cache = torch.load(cache_path, map_location="cpu")
    indices = torch.as_tensor(cache["indices"], dtype=torch.long)
    target_masks = cache.get("target_masks")
    captions = cache.get("captions", [])
    sequence_ids = cache.get("sequence_ids", [])
    window_ranges = cache.get("window_ranges", [])
    sample_ids = cache.get("sample_ids", [])
    if indices.ndim != 3:
        raise ValueError(f"cache indices must have shape (N,T,D), got {tuple(indices.shape)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_short = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(indices.shape[0]):
            target_mask = None
            if target_masks is not None:
                target_mask = torch.as_tensor(target_masks[idx], dtype=torch.bool)
            tokens = _valid_indices_from_cache_item(
                indices[idx],
                target_mask,
                max_tokens=max_tokens_per_example,
                pad_index=pad_index,
            )
            if len(tokens) < min_tokens_per_example:
                skipped_short += 1
                continue
            caption = str(captions[idx]) if idx < len(captions) else ""
            example = MotionTokenExample(
                example_id=f"{cache_path.stem}_{idx:06d}",
                caption=caption,
                indices=tokens,
                source=str(cache_path),
                sequence_id=str(sequence_ids[idx]) if idx < len(sequence_ids) else None,
                window_range=[int(x) for x in window_ranges[idx]]
                if idx < len(window_ranges) and window_ranges[idx] is not None
                else None,
                sample_ids=[str(x) for x in sample_ids[idx]]
                if idx < len(sample_ids) and sample_ids[idx] is not None
                else None,
            )
            payload = asdict(example)
            payload["indices_depth0"] = example.depth0
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            if written >= max_examples:
                break
    return {
        "cache": str(cache_path),
        "output": str(output_path),
        "examples_written": written,
        "skipped_short": skipped_short,
        "max_tokens_per_example": max_tokens_per_example,
        "min_tokens_per_example": min_tokens_per_example,
    }


def load_example_bank(path: Path) -> list[MotionTokenExample]:
    examples: list[MotionTokenExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            indices = payload.get("indices")
            if not isinstance(indices, list):
                raise ValueError(f"line {line_no}: missing indices list")
            examples.append(
                MotionTokenExample(
                    example_id=str(payload.get("example_id", f"line_{line_no:06d}")),
                    caption=str(payload.get("caption", "")),
                    indices=[[int(x) for x in row] for row in indices],
                    source=str(payload.get("source", path)),
                    sequence_id=payload.get("sequence_id"),
                    window_range=payload.get("window_range"),
                    sample_ids=payload.get("sample_ids"),
                )
            )
    return examples


def _idf_by_token(examples: list[MotionTokenExample]) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    for example in examples:
        for token in set(text_tokens(example.caption)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total = max(len(examples), 1)
    return {token: math.log((total + 1) / (freq + 0.5)) for token, freq in doc_freq.items()}


def score_example(query: str, example: MotionTokenExample, idf: dict[str, float]) -> float:
    query_tokens = text_tokens(query)
    if not query_tokens:
        return 0.0
    caption_tokens = text_tokens(example.caption)
    caption_counts: dict[str, int] = {}
    for token in caption_tokens:
        caption_counts[token] = caption_counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        if token in caption_counts:
            score += idf.get(token, 0.0) * min(caption_counts[token], 2)
    coverage = len(set(query_tokens) & set(caption_tokens)) / max(len(set(query_tokens)), 1)
    length_penalty = 1.0 / (1.0 + 0.01 * max(len(caption_tokens) - len(query_tokens), 0))
    return float((score + coverage) * length_penalty)


def retrieve_examples(
    examples: list[MotionTokenExample],
    query: str,
    *,
    top_k: int = 5,
    min_tokens: int = 2,
) -> list[tuple[MotionTokenExample, float]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    idf = _idf_by_token(examples)
    scored = [
        (example, score_example(query, example, idf))
        for example in examples
        if len(example.indices) >= min_tokens
    ]
    scored.sort(key=lambda item: (item[1], len(item[0].indices)), reverse=True)
    return scored[:top_k]


def _format_example_for_prompt(example: MotionTokenExample, max_tokens: int) -> str:
    tokens = example.indices[:max_tokens]
    return (
        f"Caption: {example.caption}\n"
        f"Tokens: {json.dumps(tokens, separators=(',', ':'))}"
    )


def build_llm_prompt(
    text: str,
    retrieved_by_segment: list[tuple[str, list[tuple[MotionTokenExample, float]]]],
    *,
    max_tokens_per_example: int = 24,
    segment_token_count: int = 25,
) -> str:
    lines = [
        "You are controlling a simulated humanoid through MoConVQ RVQ token tuples.",
        "Each motion token is a 4-integer tuple [d0,d1,d2,d3].",
        "Every integer must be in the inclusive range [0,511].",
        "Do not output prose in the final answer.",
        "Do not invent non-integer tokens.",
        "A short action should contain multiple tuples, not a single tuple.",
        "For a phrase like 'for a long time', repeat a stable locomotion subsequence.",
        "For compound prompts, concatenate sub-action sequences in order.",
        "",
        "Return only JSON with this exact schema:",
        '{"tokens":[[d0,d1,d2,d3], ...]}',
        "",
        f"Question: {text}",
        f"Target length: about {segment_token_count * max(len(retrieved_by_segment), 1)} token tuples.",
        "",
        "Retrieved examples:",
    ]
    for segment_idx, (segment, rows) in enumerate(retrieved_by_segment, start=1):
        lines.append("")
        lines.append(f"Segment {segment_idx}: {segment}")
        for rank, (example, score) in enumerate(rows, start=1):
            lines.append(f"Example {rank} score={score:.4f}")
            lines.append(_format_example_for_prompt(example, max_tokens=max_tokens_per_example))
    return "\n".join(lines).strip() + "\n"


def _find_json_payload(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty LLM response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        try:
            return json.loads(stripped[object_start : object_end + 1])
        except json.JSONDecodeError:
            pass
    list_start = stripped.find("[")
    list_end = stripped.rfind("]")
    if list_start >= 0 and list_end > list_start:
        try:
            return json.loads(stripped[list_start : list_end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("could not parse JSON payload from response") from exc
    raise ValueError("could not parse JSON payload from response")


def parse_llm_tokens(response_text: str) -> list[list[object]]:
    payload = _find_json_payload(response_text)
    if isinstance(payload, dict):
        payload = payload.get("tokens")
    if not isinstance(payload, list):
        raise ValueError("LLM response must contain a token list or {'tokens': ...}")
    return payload


def validate_token_sequence(
    raw_tokens: list[list[object]],
    *,
    rvq_depth: int = DEFAULT_RVQ_DEPTH,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_length: int = 1,
    max_length: int | None = None,
    max_consecutive_repeat: int = 5,
    repair: bool = False,
    trim_repeat_runs: bool = False,
) -> tuple[list[list[int]], dict[str, object]]:
    cleaned: list[list[int]] = []
    invalid_rows: list[dict[str, object]] = []
    clipped_values = 0
    for row_idx, row in enumerate(raw_tokens):
        if not isinstance(row, list) or len(row) != rvq_depth:
            invalid_rows.append({"row": row_idx, "reason": "wrong_depth", "value": row})
            if not repair:
                continue
            continue
        converted: list[int] = []
        row_valid = True
        for value in row:
            if isinstance(value, bool):
                row_valid = False
                break
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                row_valid = False
                break
            if int_value < 0 or int_value >= vocab_size:
                if repair:
                    int_value = min(max(int_value, 0), vocab_size - 1)
                    clipped_values += 1
                else:
                    row_valid = False
                    break
            converted.append(int_value)
        if not row_valid:
            invalid_rows.append({"row": row_idx, "reason": "invalid_value", "value": row})
            continue
        cleaned.append(converted)

    if max_length is not None:
        cleaned = cleaned[:max_length]
    repeat_repairs = 0
    if trim_repeat_runs and max_consecutive_repeat >= 1 and cleaned:
        trimmed: list[list[int]] = []
        previous: list[int] | None = None
        run_length = 0
        for token in cleaned:
            if previous is not None and token == previous:
                run_length += 1
            else:
                previous = token
                run_length = 1
            if run_length <= max_consecutive_repeat:
                trimmed.append(token)
            else:
                repeat_repairs += 1
        cleaned = trimmed

    repeat_violations = []
    if cleaned:
        run_token = cleaned[0]
        run_start = 0
        run_length = 1
        for idx in range(1, len(cleaned)):
            if cleaned[idx] == run_token:
                run_length += 1
            else:
                if run_length > max_consecutive_repeat:
                    repeat_violations.append(
                        {"start": run_start, "length": run_length, "token": run_token}
                    )
                run_token = cleaned[idx]
                run_start = idx
                run_length = 1
        if run_length > max_consecutive_repeat:
            repeat_violations.append({"start": run_start, "length": run_length, "token": run_token})

    ok = len(cleaned) >= min_length and not invalid_rows and not repeat_violations
    if repair:
        ok = len(cleaned) >= min_length and not repeat_violations
    validation = {
        "ok": bool(ok),
        "input_rows": len(raw_tokens),
        "output_rows": len(cleaned),
        "rvq_depth": rvq_depth,
        "vocab_size": vocab_size,
        "min_length": min_length,
        "max_length": max_length,
        "max_consecutive_repeat": max_consecutive_repeat,
        "invalid_rows": invalid_rows,
        "clipped_values": clipped_values,
        "trim_repeat_runs": trim_repeat_runs,
        "repeat_repairs": repeat_repairs,
        "repeat_violations": repeat_violations,
    }
    return cleaned, validation


def build_retrieval_metadata(
    examples: list[MotionTokenExample],
    text: str,
    *,
    segment_joiner: str = " then ",
    top_k: int = 5,
    min_tokens: int = 2,
) -> list[tuple[str, list[tuple[MotionTokenExample, float]]]]:
    segments = split_text_segments(text, joiner=segment_joiner)
    if not segments:
        segments = [text]
    return [
        (segment, retrieve_examples(examples, segment, top_k=top_k, min_tokens=min_tokens))
        for segment in segments
    ]


def retrieval_only_tokens(
    retrieved_by_segment: list[tuple[str, list[tuple[MotionTokenExample, float]]]],
    *,
    segment_token_count: int = 25,
) -> list[list[int]]:
    if segment_token_count < 1:
        raise ValueError("segment_token_count must be positive")
    output: list[list[int]] = []
    for _segment, rows in retrieved_by_segment:
        if not rows:
            continue
        source = rows[0][0].indices
        if not source:
            continue
        segment_tokens: list[list[int]] = []
        while len(segment_tokens) < segment_token_count:
            remaining = segment_token_count - len(segment_tokens)
            segment_tokens.extend(source[:remaining])
        output.extend(segment_tokens[:segment_token_count])
    return output


def write_tokens_json(path: Path, tokens: list[list[int]], metadata: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tokens": tokens}
    if metadata is not None:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tokens_json(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        tokens = payload.get("tokens")
    else:
        tokens = payload
    if not isinstance(tokens, list):
        raise ValueError(f"{path} does not contain a token list")
    cleaned, validation = validate_token_sequence(tokens, min_length=1, repair=False)
    if not validation["ok"]:
        raise ValueError(f"invalid token file {path}: {validation}")
    return cleaned


def rvq_tokens_to_latents(tokens: list[list[int]], base_data: Path, device: str) -> torch.Tensor:
    from Script.stage1.train_text_gpt import load_gpt_embeddings, reconstruct_latents_from_rvq_indices

    embeddings = [embedding.to(device) for embedding in load_gpt_embeddings(str(base_data))]
    indices = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    return reconstruct_latents_from_rvq_indices(indices, embeddings)


def decode_rvq_tokens_to_bvh(
    tokens: list[list[int]],
    *,
    base_data: Path,
    motion_dataset: Path | None,
    output_bvh: Path,
    gpu: int = 0,
) -> None:
    import MoConVQCore.Utils.pytorch_utils as ptu
    from Script.stage1.real_moconvq_cache import build_loaded_moconvq_agent

    ptu.init_gpu(True, gpu_id=gpu)
    agent = build_loaded_moconvq_agent(gpu=gpu, base_data=base_data, motion_dataset=motion_dataset)
    agent.eval()
    cur_embedding = rvq_tokens_to_latents(tokens, base_data=base_data, device=ptu.device)
    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

    import VclSimuBackend

    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()
    observation, info = agent.env.reset(0)

    for idx in range(dconv.shape[1]):
        obs = observation["observation"]
        action, info = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, idx],
        )
        action = ptu.to_numpy(action).flatten()
        for step_idx in range(6):
            saver.append_no_root_to_buffer()
            if step_idx == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            info = next(step_generator)
        try:
            info_ = next(step_generator)
        except StopIteration as exc:
            info_ = exc.value
        observation, _rwd, _done, info = info_

    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    saver.to_file(str(output_bvh))


def _retrieval_rows_to_json(rows: list[tuple[MotionTokenExample, float]]) -> list[dict[str, object]]:
    return [
        {
            "score": score,
            "example": {
                **asdict(example),
                "indices_depth0": example.depth0,
            },
        }
        for example, score in rows
    ]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage1 LLM in-context RVQ token planning utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-bank")
    export_parser.add_argument("--cache", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--max-examples", type=int, default=1600)
    export_parser.add_argument("--max-tokens-per-example", type=int, default=50)
    export_parser.add_argument("--min-tokens-per-example", type=int, default=3)

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--bank", required=True)
    retrieve_parser.add_argument("--query", required=True)
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    retrieve_parser.add_argument("--min-tokens", type=int, default=2)
    retrieve_parser.add_argument("--output-json", default="")

    prompt_parser = subparsers.add_parser("build-prompt")
    prompt_parser.add_argument("--bank", required=True)
    prompt_parser.add_argument("--text", required=True)
    prompt_parser.add_argument("--output-prompt", required=True)
    prompt_parser.add_argument("--output-json", default="")
    prompt_parser.add_argument("--segment-joiner", default=" then ")
    prompt_parser.add_argument("--top-k", type=int, default=5)
    prompt_parser.add_argument("--min-tokens", type=int, default=2)
    prompt_parser.add_argument("--max-tokens-per-example", type=int, default=24)
    prompt_parser.add_argument("--segment-token-count", type=int, default=25)

    validate_parser = subparsers.add_parser("validate")
    validate_source = validate_parser.add_mutually_exclusive_group(required=True)
    validate_source.add_argument("--response")
    validate_source.add_argument("--response-file")
    validate_parser.add_argument("--output-tokens", required=True)
    validate_parser.add_argument("--validation-json", required=True)
    validate_parser.add_argument("--min-length", type=int, default=1)
    validate_parser.add_argument("--max-length", type=int, default=None)
    validate_parser.add_argument("--max-consecutive-repeat", type=int, default=5)
    validate_parser.add_argument("--repair", action="store_true")
    validate_parser.add_argument("--trim-repeat-runs", action="store_true")

    retrieval_plan_parser = subparsers.add_parser("retrieval-plan")
    retrieval_plan_parser.add_argument("--bank", required=True)
    retrieval_plan_parser.add_argument("--text", required=True)
    retrieval_plan_parser.add_argument("--output-tokens", required=True)
    retrieval_plan_parser.add_argument("--validation-json", required=True)
    retrieval_plan_parser.add_argument("--segment-joiner", default=" then ")
    retrieval_plan_parser.add_argument("--top-k", type=int, default=5)
    retrieval_plan_parser.add_argument("--min-tokens", type=int, default=2)
    retrieval_plan_parser.add_argument("--segment-token-count", type=int, default=25)
    retrieval_plan_parser.add_argument("--max-consecutive-repeat", type=int, default=5)
    retrieval_plan_parser.add_argument("--trim-repeat-runs", action="store_true")

    decode_parser = subparsers.add_parser("decode-bvh")
    decode_parser.add_argument("--tokens", required=True)
    decode_parser.add_argument("--base-data", default="moconvq_base.data")
    decode_parser.add_argument("--motion-dataset", default="")
    decode_parser.add_argument("--output-bvh", required=True)
    decode_parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "export-bank":
        summary = export_example_bank_from_cache(
            Path(args.cache),
            Path(args.output),
            max_examples=args.max_examples,
            max_tokens_per_example=args.max_tokens_per_example,
            min_tokens_per_example=args.min_tokens_per_example,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "retrieve":
        examples = load_example_bank(Path(args.bank))
        rows = retrieve_examples(examples, args.query, top_k=args.top_k, min_tokens=args.min_tokens)
        payload = {"query": args.query, "results": _retrieval_rows_to_json(rows)}
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    elif args.command == "build-prompt":
        examples = load_example_bank(Path(args.bank))
        retrieved = build_retrieval_metadata(
            examples,
            args.text,
            segment_joiner=args.segment_joiner,
            top_k=args.top_k,
            min_tokens=args.min_tokens,
        )
        prompt = build_llm_prompt(
            args.text,
            retrieved,
            max_tokens_per_example=args.max_tokens_per_example,
            segment_token_count=args.segment_token_count,
        )
        output_prompt = Path(args.output_prompt)
        output_prompt.parent.mkdir(parents=True, exist_ok=True)
        output_prompt.write_text(prompt, encoding="utf-8")
        payload = {
            "text": args.text,
            "segments": [
                {"segment": segment, "results": _retrieval_rows_to_json(rows)}
                for segment, rows in retrieved
            ],
        }
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output_prompt": str(output_prompt), "segments": len(retrieved)}, indent=2))
    elif args.command == "validate":
        response = args.response if args.response is not None else Path(args.response_file).read_text(encoding="utf-8")
        raw_tokens = parse_llm_tokens(response)
        cleaned, validation = validate_token_sequence(
            raw_tokens,
            min_length=args.min_length,
            max_length=args.max_length,
            max_consecutive_repeat=args.max_consecutive_repeat,
            repair=args.repair,
            trim_repeat_runs=args.trim_repeat_runs,
        )
        write_tokens_json(Path(args.output_tokens), cleaned, metadata={"source": "llm_response"})
        Path(args.validation_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.validation_json).write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps(validation, indent=2))
    elif args.command == "retrieval-plan":
        examples = load_example_bank(Path(args.bank))
        retrieved = build_retrieval_metadata(
            examples,
            args.text,
            segment_joiner=args.segment_joiner,
            top_k=args.top_k,
            min_tokens=args.min_tokens,
        )
        tokens = retrieval_only_tokens(retrieved, segment_token_count=args.segment_token_count)
        cleaned, validation = validate_token_sequence(
            tokens,
            min_length=1,
            max_consecutive_repeat=args.max_consecutive_repeat,
            repair=False,
            trim_repeat_runs=args.trim_repeat_runs,
        )
        metadata = {
            "source": "retrieval_only",
            "text": args.text,
            "segment_token_count": args.segment_token_count,
            "trim_repeat_runs": args.trim_repeat_runs,
            "max_consecutive_repeat": args.max_consecutive_repeat,
            "segments": [
                {
                    "segment": segment,
                    "selected_example": rows[0][0].example_id if rows else None,
                    "selected_caption": rows[0][0].caption if rows else None,
                    "score": rows[0][1] if rows else None,
                }
                for segment, rows in retrieved
            ],
        }
        write_tokens_json(Path(args.output_tokens), cleaned, metadata=metadata)
        Path(args.validation_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.validation_json).write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps({"tokens": len(cleaned), "validation": validation}, indent=2))
    elif args.command == "decode-bvh":
        tokens = load_tokens_json(Path(args.tokens))
        decode_rvq_tokens_to_bvh(
            tokens,
            base_data=Path(args.base_data),
            motion_dataset=Path(args.motion_dataset) if args.motion_dataset else None,
            output_bvh=Path(args.output_bvh),
            gpu=args.gpu,
        )
        print(json.dumps({"output_bvh": args.output_bvh, "tokens": len(tokens)}, indent=2))


if __name__ == "__main__":
    main()
