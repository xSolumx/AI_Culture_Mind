from __future__ import annotations

import unittest

import torch

from intertwiner_schurscan_equivariant_identification import (
    dataset_from_sequences,
    fit_generic_bilinear,
    make_instance,
    prediction_metrics,
    random_subspace_dataset,
    scan_parity,
    transform_dataset,
)


class IntertwinerSchurScanEquivariantIdentificationTests(unittest.TestCase):
    def test_closed_form_generic_fit_interpolates_seen_support(self) -> None:
        instance = make_instance("spin8_triality")
        generator = torch.Generator().manual_seed(4)
        dataset = random_subspace_dataset(
            instance, count=64, length=8, generator=generator
        )
        fitted, design = fit_generic_bilinear(
            dataset.bilinear_features,
            dataset.target,
            output_dimension=instance.output_dimension,
            first_dimension=instance.first_dimension,
            second_dimension=instance.second_dimension,
        )
        prediction = dataset.bilinear_features @ fitted.flatten(1).T
        metrics = prediction_metrics(prediction, dataset.target)
        self.assertEqual(design["rank"], 16)
        self.assertLess(metrics["mean_relative_squared_error"], 1e-20)

    def test_group_transforms_preserve_bilinear_teacher(self) -> None:
        for name in ("spin8_triality", "so3_cross_product"):
            with self.subTest(name=name):
                instance = make_instance(name)
                generator = torch.Generator().manual_seed(7)
                dataset = random_subspace_dataset(
                    instance, count=8, length=11, generator=generator
                )
                transformed, error = transform_dataset(
                    dataset, instance, generator=generator
                )
                recomputed = dataset_from_sequences(
                    transformed.first, transformed.second, instance.beta
                )
                self.assertLess(error, 1e-11)
                torch.testing.assert_close(
                    transformed.target, recomputed.target, rtol=1e-11, atol=1e-11
                )

    def test_scan_recurrence_and_closed_form_agree(self) -> None:
        for name in ("spin8_triality", "so3_cross_product"):
            with self.subTest(name=name):
                report = scan_parity(
                    make_instance(name),
                    generator=torch.Generator().manual_seed(11),
                )
                errors = [
                    float(value)
                    for key, value in report.items()
                    if key.endswith("error")
                ]
                self.assertLess(max(errors), 1e-10)


if __name__ == "__main__":
    unittest.main()
