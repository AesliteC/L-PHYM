import re
import subprocess
import unittest
from pathlib import Path


class Stage1RepositoryHygieneTests(unittest.TestCase):
    def test_agent_private_docs_are_not_tracked(self):
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        tracked = result.stdout.decode("utf-8").split("\0")
        forbidden = re.compile(
            r"(^|/)(AGENT\.md|AGENTS\.md|CODEX\.md|CLAUDE\.md|\.codex(/|$)|\.claude(/|$))",
            re.IGNORECASE,
        )
        matches = [path for path in tracked if forbidden.search(path)]

        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
