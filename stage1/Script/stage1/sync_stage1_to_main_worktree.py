from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


EXCLUDES = (
    ".git",
    ".vscode",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    ".venv",
    ".pyenv",
    ".conda",
    "out",
    "build",
    "stage1_artifacts",
    "*.h5",
    "*.pth",
    "*.data",
    "AGENT.md",
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    ".codex",
    ".claude",
    "midterm-report",
    "midterm_figures",
    "request.txt",
)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=cwd, check=True, text=True)


def _git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def sync_stage1_to_main_worktree(
    *,
    true_workdir: Path,
    main_worktree: Path,
    delete: bool,
    dry_run: bool,
) -> None:
    true_workdir = true_workdir.resolve()
    main_worktree = main_worktree.resolve()
    stage1_dir = main_worktree / "stage1"
    if not true_workdir.exists():
        raise FileNotFoundError(f"true workdir not found: {true_workdir}")
    if not stage1_dir.exists():
        raise FileNotFoundError(f"main worktree stage1 dir not found: {stage1_dir}")
    branch = _git_output(["branch", "--show-current"], main_worktree)
    if branch != "main":
        raise RuntimeError(f"{main_worktree} must be on branch main, got {branch!r}")

    cmd = ["rsync", "-a"]
    if delete:
        cmd.append("--delete")
    for pattern in EXCLUDES:
        cmd.extend(["--exclude", pattern])
    cmd.extend([f"{true_workdir}/", f"{stage1_dir}/"])
    _run(cmd, dry_run=dry_run)

    if not dry_run:
        _run(["git", "status", "--short", "--branch"], cwd=main_worktree)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--true-workdir",
        default="/home/chenjie/cc/robotics/MoConVQ",
        help="Stage1 working tree used for experiments.",
    )
    parser.add_argument(
        "--main-worktree",
        default="/home/chenjie/cc/robotics/MoConVQ-main",
        help="Checkout of origin/main whose stage1/ folder is pushed to GitHub.",
    )
    parser.add_argument("--delete", action="store_true", help="Also delete stale files in main/stage1.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    sync_stage1_to_main_worktree(
        true_workdir=Path(args.true_workdir),
        main_worktree=Path(args.main_worktree),
        delete=args.delete,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
