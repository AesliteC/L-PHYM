import unittest


class Stage1BVHCharacterCacheTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
