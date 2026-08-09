import unittest

from p1train.flair_shuffle import FlairCropRecord, build_density_matched_derangement


class FlairShuffleTest(unittest.TestCase):
    def _records(self):
        return tuple(
            FlairCropRecord(
                update=index,
                source_index=index,
                source_identifier=f"D{index // 3}/crop_{index}",
                domain=f"D{index // 3}",
                id10_pixels=value,
                crop_pixels=100,
            )
            for index, value in enumerate((1, 3, 5, 10, 12, 14, 20, 22, 24))
        )

    def test_is_deterministic_and_preserves_density_multiset(self):
        records = self._records()
        first = build_density_matched_derangement(records, seed=19)
        second = build_density_matched_derangement(records, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first.permutation), list(range(len(records))))
        self.assertEqual(
            sorted(records[index].id10_pixels for index in first.permutation),
            sorted(record.id10_pixels for record in records),
        )
        self.assertGreaterEqual(first.max_abs_density_difference, first.mean_abs_density_difference)
        self.assertGreaterEqual(first.mean_abs_density_difference, 0.0)

    def test_no_native_pair_and_domain_is_preserved(self):
        records = self._records()
        plan = build_density_matched_derangement(records, seed=37)
        for image_index, mask_index in enumerate(plan.permutation):
            self.assertNotEqual(records[image_index].source_identifier, records[mask_index].source_identifier)
            self.assertEqual(records[image_index].domain, records[mask_index].domain)

    def test_singleton_density_bins_are_recorded_as_merged(self):
        records = self._records()
        plan = build_density_matched_derangement(records, seed=73, density_bin_width=0.01)
        self.assertTrue(plan.merged_singleton_bins)
        self.assertTrue(all(len(block) in (2, 3) for block in plan.blocks))

    def test_rejects_a_domain_without_a_derangement(self):
        records = (
            FlairCropRecord(0, 0, "D0/crop_0", "D0", 2, 100),
            FlairCropRecord(1, 1, "D1/crop_1", "D1", 3, 100),
        )
        with self.assertRaises(ValueError):
            build_density_matched_derangement(records, seed=19)


if __name__ == "__main__":
    unittest.main()
