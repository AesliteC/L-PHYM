from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path


class Stage1PrepareT2MEvaluatorAssetsTests(unittest.TestCase):
    def test_check_reports_missing_assets_and_sources(self):
        from Script.stage1.prepare_t2m_evaluator_assets import check_t2m_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = check_t2m_assets(Path(tmpdir))

        self.assertFalse(payload["ready"])
        self.assertIn("checkpoints/t2m/text_mot_match/model/finest.tar", payload["missing_assets"])
        self.assertIn("models/evaluator_wrapper.py", payload["missing_sources"])

    def test_check_detects_ready_layout(self):
        from Script.stage1.prepare_t2m_evaluator_assets import REQUIRED_T2M_ASSETS, T2M_SOURCE_HINTS, check_t2m_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative in (*REQUIRED_T2M_ASSETS, *T2M_SOURCE_HINTS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            payload = check_t2m_assets(root)

        self.assertTrue(payload["ready"])
        self.assertEqual(payload["missing_assets"], [])
        self.assertEqual(payload["missing_sources"], [])

    def test_unpack_archives_extracts_expected_layout(self):
        from Script.stage1.prepare_t2m_evaluator_assets import unpack_archives

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t2m_zip = root / "t2m.zip"
            glove_zip = root / "glove.zip"
            with zipfile.ZipFile(t2m_zip, "w") as archive:
                archive.writestr("checkpoints/t2m/text_mot_match/model/finest.tar", "")
                archive.writestr("checkpoints/t2m/text_mot_match/opt.txt", "")
            with zipfile.ZipFile(glove_zip, "w") as archive:
                archive.writestr("glove/our_vab_data.npy", "")
                archive.writestr("glove/our_vab_words.pkl", "")

            payload = unpack_archives(root / "assets", t2m_zip=t2m_zip, glove_zip=glove_zip)

        self.assertFalse(payload["status"]["ready"])
        self.assertTrue(payload["status"]["assets_ready"])
        self.assertFalse(payload["status"]["sources_ready"])

    def test_download_commands_include_proxy_and_readiness_check(self):
        from Script.stage1.prepare_t2m_evaluator_assets import download_commands

        commands = download_commands(Path("/tmp/assets"), python="/env/python")
        text = "\n".join(commands)

        self.assertIn("http_proxy", text)
        self.assertIn("/env/python -m gdown", text)
        self.assertIn("--copy-sources", text)
        self.assertIn("check_evaluation_readiness.py", text)

    def test_copy_evaluator_sources(self):
        from Script.stage1.prepare_t2m_evaluator_assets import T2M_SOURCE_HINTS, copy_evaluator_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            output = root / "output"
            for relative in T2M_SOURCE_HINTS:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")

            payload = copy_evaluator_sources(source, output)

        self.assertEqual(sorted(payload["copied_sources"]), sorted(T2M_SOURCE_HINTS))
        self.assertEqual(payload["missing_sources"], [])
        self.assertTrue(payload["status"]["sources_ready"])


if __name__ == "__main__":
    unittest.main()
