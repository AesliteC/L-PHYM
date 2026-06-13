import unittest
from pathlib import Path
import tempfile
from unittest import mock


class Stage1BVHCharacterCacheTests(unittest.TestCase):
    def test_direct_script_execution_prefers_own_repo_root(self):
        import Script.stage1.build_bvh_character_gpt_cache as cache_builder

        expected_root = Path(cache_builder.__file__).resolve().parents[2]
        old_path = list(cache_builder.sys.path)
        try:
            cache_builder.sys.path[:] = ["/tmp/other_checkout"] + [
                path for path in old_path if Path(path or ".").resolve() != expected_root
            ]
            cache_builder._ensure_own_repo_root_on_path(package="")
            self.assertEqual(Path(cache_builder.sys.path[0]).resolve(), expected_root)
        finally:
            cache_builder.sys.path[:] = old_path

    def test_parse_bvh_specs_accepts_path_caption_pairs(self):
        from Script.stage1.build_bvh_character_gpt_cache import parse_bvh_specs

        specs = parse_bvh_specs(["walk.bvh=a person walks", "kick.bvh=a person kicks"])

        self.assertEqual(str(specs[0][0]), "walk.bvh")
        self.assertEqual(specs[0][1], "a person walks")
        self.assertEqual(str(specs[1][0]), "kick.bvh")
        self.assertEqual(specs[1][1], "a person kicks")

    def test_parse_bvh_specs_rejects_missing_caption_separator(self):
        from Script.stage1.build_bvh_character_gpt_cache import parse_bvh_specs

        with self.assertRaises(ValueError):
            parse_bvh_specs(["walk.bvh"])

    def test_specs_from_quality_summary_uses_accepted_rows_by_default(self):
        import json

        from Script.stage1.build_bvh_character_gpt_cache import specs_from_quality_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "quality.json"
            summary.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"path": "good.bvh", "caption": "a good motion", "accepted": True},
                            {"path": "bad.bvh", "caption": "a bad motion", "accepted": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            specs = specs_from_quality_summary(summary)
            self.assertEqual(specs, [(Path("good.bvh"), "a good motion")])

            all_specs = specs_from_quality_summary(summary, accepted_only=False)
            self.assertEqual([path for path, _caption in all_specs], [Path("good.bvh"), Path("bad.bvh")])

    def test_bvh_specs_from_quality_summary_uses_export_metadata_when_available(self):
        import json

        from Script.stage1.build_bvh_character_gpt_cache import bvh_specs_from_quality_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            quality = tmp / "quality.json"
            export = tmp / "export.json"
            quality.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "label": "seq_000",
                                "path": "/tmp/seq_000.bvh",
                                "caption": "",
                                "accepted": True,
                                "sample_ids": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            export.write_text(
                json.dumps(
                    {
                        "exports": [
                            {
                                "sample_id": "seq_000",
                                "sequence_id": "seq_000",
                                "output_bvh": "/tmp/seq_000.bvh",
                                "caption": "a person sits then waves",
                                "sample_ids": ["M000001", "M000002"],
                                "clip_captions": ["a person sits", "a person waves"],
                                "clip_boundaries": [[0, 3], [3, 6]],
                                "transition_forced": [False],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            specs = bvh_specs_from_quality_summary(quality, export_summary=export)

        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec.caption, "a person sits then waves")
        self.assertEqual(spec.sample_ids, ("M000001", "M000002"))
        self.assertEqual(spec.clip_captions, ("a person sits", "a person waves"))
        self.assertEqual(spec.clip_boundaries, ((0, 3), (3, 6)))
        self.assertEqual(spec.transition_forced, (False,))

    def test_main_forwards_motion_dataset_to_agent_loader(self):
        import torch

        from Script.stage1 import build_bvh_character_gpt_cache as cache_builder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bvh = tmp / "walk.bvh"
            bvh.write_text("HIERARCHY\nMOTION\nFrames: 0\nFrame Time: 0.05\n", encoding="utf-8")
            motion_dataset = tmp / "simple_motion_data.h5"
            motion_dataset.write_bytes(b"")
            output = tmp / "cache.pt"
            observation = tmp / "obs.h5"
            summary = tmp / "summary.json"
            fake_cache = {
                "indices": torch.zeros((1, 2, 4), dtype=torch.long),
                "latents": torch.zeros((1, 2, 768), dtype=torch.float32),
                "text_features": torch.zeros((1, 4, 1024), dtype=torch.float32),
                "text_masks": torch.zeros((1, 4), dtype=torch.bool),
                "captions": ["a person walks"],
                "sequence_ids": ["walk"],
                "target_mask": torch.ones((1, 2), dtype=torch.bool),
                "config": {},
            }

            with mock.patch.object(cache_builder, "build_loaded_moconvq_agent", return_value=object()) as loader:
                with mock.patch.object(cache_builder, "build_t5_text_encoder", return_value=object()):
                    with mock.patch.object(cache_builder, "build_bvh_character_cache", return_value=fake_cache):
                        cache_builder.main(
                            [
                                "--bvh",
                                f"{bvh}=a person walks",
                                "--base-data",
                                "base.data",
                                "--motion-dataset",
                                str(motion_dataset),
                                "--output",
                                str(output),
                                "--observation-h5",
                                str(observation),
                                "--summary",
                                str(summary),
                            ]
                        )

            loader.assert_called_once()
            self.assertEqual(loader.call_args.kwargs["motion_dataset"], motion_dataset)
            self.assertTrue(output.exists())
            self.assertTrue(summary.exists())

    def test_segment_prefix_cache_uses_local_clip_captions(self):
        import numpy as np
        import torch

        from Script.stage1 import build_bvh_character_gpt_cache as cache_builder

        spec = cache_builder.BVHSpec(
            path=Path("demo.bvh"),
            caption="sit then wave",
            sample_ids=("M000001", "M000002"),
            clip_captions=("a person sits", "a person waves"),
            clip_boundaries=((0, 3), (3, 6)),
        )
        rows = [{"key": "0000_demo", "observation": np.zeros((6, 323), dtype=np.float32), "spec": spec}]

        def fake_encode_observation_with_agent(agent, observation, rvq_depth=4):
            latent = np.arange(6 * 768, dtype=np.float32).reshape(6, 768)
            indices = np.zeros((6, rvq_depth), dtype=np.int64)
            return latent, indices

        calls: list[str] = []

        def fake_text_encoder(texts):
            calls.extend(texts)
            value = float(len(calls))
            return np.full((len(texts), 4, 1024), value, dtype=np.float32), np.zeros((len(texts), 4), dtype=bool)

        old_encode = cache_builder.encode_observation_with_agent
        cache_builder.encode_observation_with_agent = fake_encode_observation_with_agent
        try:
            cache = cache_builder.build_cache_from_bvh_observations(
                rows=rows,
                agent=object(),
                text_encoder=fake_text_encoder,
                window_size=3,
                window_stride=3,
                rvq_depth=4,
                caption_mode="window",
                window_policy="clip",
                sample_mode="segment_prefix",
                prefix_size=0,
            )
        finally:
            cache_builder.encode_observation_with_agent = old_encode

        self.assertEqual(calls, ["a person sits", "a person waves"])
        self.assertEqual(cache["captions"], ["a person sits", "a person waves"])
        self.assertEqual(cache["sample_ids"], [["M000001", "M000002"], ["M000001", "M000002"]])
        self.assertTrue(torch.equal(cache["segment_idxs"], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(cache["num_segments"], torch.tensor([2, 2])))


if __name__ == "__main__":
    unittest.main()
