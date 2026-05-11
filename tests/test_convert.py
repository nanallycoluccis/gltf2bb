from __future__ import annotations

import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path

from src.config import FaceFeatureProtectionConfig
from src.convert import convert_model, protect_explicit_face_feature_visibility, split_hair_part_faces
from src.conversion.types import BBoxAccumulator, PartKey, SplitFace
from src.inspect import main


class ConvertModelTest(unittest.TestCase):
    def test_writes_one_bbox_cube_per_assigned_bone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "two_bone_cuboids.gltf"
            output_path = Path(temp_dir) / "out" / "model.bbmodel"
            write_two_bone_cuboid_fixture(model_path)

            result = convert_model(model_path, output_path, target_height=4.0)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 2)
        self.assertEqual(data["meta"]["model_format"], "free")
        self.assertEqual(data["resolution"], {"width": 16, "height": 16})

        elements = {element["name"]: element for element in data["elements"]}
        self.assertEqual(set(elements), {"root_joint_cube", "child_joint_cube"})
        self.assertEqual(elements["root_joint_cube"]["from"], [-2.0, 0.0, 0.0])
        self.assertEqual(elements["root_joint_cube"]["to"], [0.0, 2.0, 0.0])
        self.assertEqual(elements["child_joint_cube"]["from"], [0.0, 2.0, 0.0])
        self.assertEqual(elements["child_joint_cube"]["to"], [2.0, 4.0, 0.0])

        groups = {group["name"]: group for group in data["groups"]}
        self.assertEqual(groups["root_joint"]["origin"], [0.0, 0.0, 0.0])
        self.assertEqual(groups["child_joint"]["origin"], [0.0, 2.0, 0.0])

    def test_cli_convert_command_writes_bbmodel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "two_bone_cuboids.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_two_bone_cuboid_fixture(model_path)

            exit_code = main(
                [
                    "convert",
                    str(model_path),
                    "-o",
                    str(output_path),
                    "--mode",
                    "cuboid",
                    "--target-height",
                    "4",
                ]
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(data["elements"]), 2)

    def test_expands_nonzero_sub_minimum_thickness_from_source_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "thin_nonzero_triangle.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_thin_nonzero_triangle_fixture(model_path)

            convert_model(model_path, output_path, target_height=4.0)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        element = data["elements"][0]
        self.assertEqual(element["from"], [-0.005, 0.0, -0.005])
        self.assertEqual(element["to"], [0.005, 4.0, 0.005])

    def test_collapses_near_planar_source_geometry_to_zero_thickness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "near_planar_triangle.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_near_planar_triangle_fixture(model_path)

            convert_model(model_path, output_path, target_height=4.0)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        element = data["elements"][0]
        self.assertEqual(element["from"], [-2.0, 0.0, 0.0])
        self.assertEqual(element["to"], [2.0, 4.0, 0.0])

    def test_cli_convert_command_accepts_hybrid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hybrid_body_hair.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_hybrid_body_hair_fixture(model_path)

            exit_code = main(
                [
                    "convert",
                    str(model_path),
                    "-o",
                    str(output_path),
                    "--mode",
                    "hybrid",
                    "--target-height",
                    "4",
                ]
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {element["name"] for element in data["elements"]},
            {"body_cube", "hair_front_1_cube", "hair_front_2_cube"},
        )

    def test_cli_convert_command_accepts_complex_split_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "complex_head.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_head_complex_fixture(model_path)

            exit_code = main(
                [
                    "convert",
                    str(model_path),
                    "-o",
                    str(output_path),
                    "--target-height",
                    "4",
                    "--complex-split",
                    "head",
                ]
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual({element["name"] for element in data["elements"]}, {"head_core_cube", "hair_front_cube", "hair_back_cube"})

    def test_preset_merges_helper_bone_before_writing_bbmodel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "helper_bone_cuboid.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_helper_merge_cuboid_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=2.0,
                preset="mmd_humanoid",
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.preset, "mmd_humanoid")
        self.assertEqual(len(result.cubes), 1)
        self.assertEqual(len(result.bone_resolution.merged_to_parent), 1)
        self.assertEqual(result.empty_bones, 1)
        self.assertEqual({group["name"] for group in data["groups"]}, {"root_joint", "child_joint"})
        self.assertEqual([element["name"] for element in data["elements"]], ["root_joint_cube"])
        self.assertEqual(report["totals"]["merged_bones"], 1)
        self.assertIn("small_cubes", report["totals"])

    def test_convert_report_includes_quality_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "quality_diagnostics.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_quality_diagnostics_fixture(model_path)

            convert_model(model_path, output_path, target_height=4.0, report_path=report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        quality = report["quality"]
        self.assertEqual(quality["skipped_unskinned_meshes_summary"]["count"], 1)
        self.assertEqual(quality["skipped_unskinned_meshes_summary"]["node_indices"], [2])
        self.assertEqual(quality["skipped_unskinned_meshes_summary"]["mesh_indices"], [0])

        largest = quality["largest_cubes"][0]
        self.assertEqual(largest["name"], "root_joint_cube")
        self.assertEqual(largest["owner_bone_name"], "root_joint")
        self.assertEqual(largest["dimensions"], [40.0, 4.0, 0.0])
        self.assertIsNone(largest["rotation_source"])

        elongated = quality["unrotated_elongated_cubes"][0]
        self.assertEqual(elongated["name"], "root_joint_cube")
        self.assertIn("no rotation", elongated["reason"])
        self.assertEqual(quality["tiny_fragment_cubes"], [])

    def test_unskinned_meshes_can_assign_by_node_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "unskinned_parent.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_unskinned_parent_fixture(model_path)
            config_path.write_text(
                json.dumps({"unskinned_meshes": {"enabled": True, "strategy": "node_parent"}}),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.assigned_unskinned_meshes), 1)
        self.assertEqual(result.skipped_unskinned_meshes, [])
        self.assertEqual(
            {element["name"] for element in data["elements"]},
            {"root_joint_cube", "child_joint_unskinned_3_1_0_cube"},
        )

        assigned = report["unskinned_meshes"]["assigned"][0]
        self.assertTrue(report["unskinned_meshes"]["enabled"])
        self.assertEqual(assigned["owner_bone_name"], "child_joint")
        self.assertEqual(assigned["part_name"], "child_joint_unskinned_3_1_0")
        self.assertEqual(assigned["reason"], "node_parent")
        self.assertEqual(report["totals"]["assigned_unskinned_meshes"], 1)
        self.assertEqual(report["quality"]["skipped_unskinned_meshes_summary"]["count"], 0)

    def test_unskinned_meshes_skip_default_ignored_helper_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "unskinned_helper_material.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_unskinned_parent_fixture(model_path, ignored_static_material=True)
            config_path.write_text(
                json.dumps({"unskinned_meshes": {"enabled": True, "strategy": "node_parent_then_nearest"}}),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.assigned_unskinned_meshes, [])
        self.assertEqual(len(result.skipped_unskinned_meshes), 1)
        self.assertEqual([element["name"] for element in data["elements"]], ["root_joint_cube"])

        skipped = report["unskinned_meshes"]["skipped"][0]
        self.assertEqual(skipped["primitive_index"], 0)
        self.assertEqual(skipped["reason"], "ignored_material_name_contains:mmd_tools_rigid")
        self.assertEqual(report["totals"]["assigned_unskinned_meshes"], 0)
        self.assertEqual(report["totals"]["skipped_unskinned_meshes"], 1)

    def test_complex_split_head_outputs_subpart_cubes_under_head_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "complex_head.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_head_complex_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        self.assertEqual(result.empty_bones, 1)
        self.assertEqual(len(result.cubes), 3)
        self.assertEqual(element_names, {"head_core_cube", "hair_front_cube", "hair_back_cube"})
        self.assertEqual(report["totals"]["complex_split_bones"], 1)

        split = report["complex_split"][0]
        self.assertEqual(split["bone_name"], "head")
        self.assertEqual(
            {subpart["name"] for subpart in split["subparts"]},
            {"head_core", "hair_front", "hair_back"},
        )

        groups = {group["name"]: group for group in data["groups"]}
        cube_uuids = {element["uuid"] for element in data["elements"]}
        head_entry = find_outliner_entry(data["outliner"], groups["head"]["uuid"])
        self.assertIsNotNone(head_entry)
        self.assertEqual(set(head_entry["children"]), cube_uuids)

    def test_large_head_core_splits_into_multiple_body_cubes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "large_head_core.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_large_head_core_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        self.assertGreaterEqual(len(result.cubes), 2)
        self.assertNotIn("head_core_cube", element_names)
        self.assertTrue(all(name.startswith("head_core_") for name in element_names))

        head_split = report["complex_split"][0]
        self.assertEqual(head_split["bone_name"], "head")
        self.assertGreaterEqual(len(head_split["subparts"]), 2)
        self.assertTrue(any(subpart["name"].startswith("head_core_") for subpart in head_split["subparts"]))

    def test_large_head_core_with_face_features_does_not_create_face_recess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "large_head_core_face_features.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_large_head_core_with_face_features_fixture(model_path)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        self.assertIn("head_core_front_lower_cube", elements)
        self.assertIn("head_core_front_upper_cube", elements)
        self.assertFalse(any(name.startswith("head_core_front") and "face_recess" in name for name in elements))
        self.assertFalse(any(name.startswith("head_core_front") and name.endswith("_face_cube") for name in elements))

        subparts = {subpart["name"] for subpart in report["complex_split"][0]["subparts"]}
        self.assertFalse(any(name.startswith("head_core_front") and "face_recess" in name for name in subparts))
        self.assertFalse(any(name.startswith("head_core_front") and name.endswith("_face") for name in subparts))

    def test_head_face_features_and_front_hair_keep_source_planes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "large_head_core_face_features.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_large_head_core_with_face_features_fixture(model_path, include_front_hair=True)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        upper_hair = [element for name, element in elements.items() if name.startswith("hair_front") and "upper" in name]
        self.assertTrue(upper_hair)

        face_features = [
            element
            for name, element in elements.items()
            if name.startswith(("eye", "brow", "mouth"))
        ]
        self.assertTrue(face_features)
        for element in face_features:
            self.assertEqual(element["from"][2], element["to"][2])

        feature_max_y = max(
            element["to"][1]
            for element in face_features
        )
        self.assertLess(min(element["from"][1] for element in upper_hair), feature_max_y)

    def test_explicit_eye_bone_suppresses_duplicate_head_face_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "explicit_eye_head.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_explicit_eye_head_fixture(model_path)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        self.assertIn("目.R_cube", elements)
        self.assertIn("目.L_cube", elements)
        self.assertFalse({"eye_l_cube", "eye_r_cube", "mouth_cube"} & set(elements))
        self.assertFalse(any(name.startswith("head_core_front") and name.endswith("_face_cube") for name in elements))

        subparts = {subpart["name"] for subpart in report["complex_split"][0]["subparts"]}
        self.assertFalse({"eye_l", "eye_r", "mouth"} & subparts)
        self.assertFalse(any(name.startswith("head_core_front") and name.endswith("_face") for name in subparts))

    def test_explicit_face_features_limit_head_core_and_front_hair_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "explicit_eye_front_hair.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_explicit_eye_head_fixture(model_path, include_front_hair=True)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        eye_elements = [element for name, element in elements.items() if name.startswith("目.")]
        head_front_elements = [element for name, element in elements.items() if name.startswith("head_core_front")]
        hair_front_elements = [element for name, element in elements.items() if name.startswith("hair_front")]

        self.assertTrue(eye_elements)
        self.assertTrue(head_front_elements)
        self.assertTrue(hair_front_elements)

        eye_min_z = min(element["from"][2] for element in eye_elements)
        eye_max_y = max(element["to"][1] for element in eye_elements)
        self.assertTrue(all(element["to"][2] < eye_min_z for element in head_front_elements))
        self.assertIn("head_core_front_lower_cube", elements)
        self.assertNotIn("head_core_front_lower_left_cube", elements)
        self.assertTrue(all(element["from"][1] > eye_max_y for element in hair_front_elements))
        protection_report = report["face_feature_protection"]
        self.assertTrue(protection_report["enabled"])
        self.assertEqual(protection_report["min_faces"], 32)
        self.assertFalse(
            any(action["action"] == "split_head_core_around_features" for action in protection_report["actions"])
        )
        self.assertTrue(
            any(action["action"] == "raise_hair_front_above_features" for action in protection_report["actions"])
        )

    def test_face_feature_protection_can_disable_front_hair_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "explicit_eye_front_hair.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_explicit_eye_head_fixture(model_path, include_front_hair=True)
            config_path.write_text(
                json.dumps(
                    {
                        "complex_split": {"enabled": True, "bones": ["head"]},
                        "face_feature_protection": {"protect_hair_front": False},
                    }
                ),
                encoding="utf-8",
            )

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        eye_max_y = max(element["to"][1] for name, element in elements.items() if name.startswith("目."))
        hair_front_elements = [element for name, element in elements.items() if name.startswith("hair_front")]

        self.assertTrue(hair_front_elements)
        self.assertTrue(any(element["from"][1] < eye_max_y for element in hair_front_elements))
        protection_report = report["face_feature_protection"]
        self.assertFalse(protection_report["protect_hair_front"])
        self.assertFalse(any(action["action"] == "raise_hair_front_above_features" for action in protection_report["actions"]))

    def test_explicit_face_features_do_not_expand_head_core_toward_face(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "explicit_eye_head.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_explicit_eye_head_fixture(model_path)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        eye_min_z = min(element["from"][2] for name, element in elements.items() if name.startswith("目."))
        head_front_elements = [element for name, element in elements.items() if name.startswith("head_core_front")]

        self.assertTrue(head_front_elements)
        self.assertTrue(all(element["to"][2] < eye_min_z for element in head_front_elements))
        self.assertFalse(
            any(action["action"] == "clamp_head_core_behind_features" for action in report["face_feature_protection"]["actions"])
        )

    def test_face_feature_protection_clamps_back_head_core_bucket_that_reaches_face(self) -> None:
        accumulator = BBoxAccumulator(min_xyz=[-1.0, 0.0, -0.5], max_xyz=[1.0, 1.0, 1.4], faces=10)
        accumulators = {PartKey(1, "head_core_back_lower"): accumulator}

        actions = protect_explicit_face_feature_visibility(
            1,
            accumulators,
            None,
            ([-0.5, 0.2, 1.0], [0.5, 0.8, 1.2]),
            ([-1.0, 0.0, -0.5], [1.0, 1.0, 1.4]),
            1,
            FaceFeatureProtectionConfig(),
            {},
        )

        self.assertEqual([action.action for action in actions], ["clamp_head_core_behind_features"])
        self.assertLess(accumulator.max_xyz[2], 1.0)

    def test_complex_split_merges_tiny_connected_components_to_nearest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "head_tiny_component.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_head_tiny_component_fixture(model_path)
            config_path.write_text(
                json.dumps(
                    {
                        "complex_split": {
                            "enabled": True,
                            "bones": ["head"],
                            "connected_components": {"min_faces": 2},
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual([element["name"] for element in data["elements"]], ["head_core_cube"])
        self.assertEqual(result.complex_split[0].merged_tiny_components, 1)
        self.assertEqual(report["complex_split"][0]["merged_tiny_components"], 1)
        self.assertEqual(report["complex_split"][0]["deleted_tiny_components"], 0)

        subparts = {subpart["name"]: subpart for subpart in report["complex_split"][0]["subparts"]}
        self.assertEqual(set(subparts), {"head_core"})
        self.assertEqual(subparts["head_core"]["faces"], 3)

    def test_complex_split_keeps_separate_hair_components_as_editable_cubes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hair_strands.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_head_multiple_hair_components_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        self.assertEqual(len(result.cubes), 3)
        self.assertEqual(element_names, {"head_core_cube", "hair_front_cube", "hair_front_2_cube"})
        self.assertEqual(
            {subpart["name"] for subpart in report["complex_split"][0]["subparts"]},
            {"head_core", "hair_front", "hair_front_2"},
        )

    def test_complex_split_can_target_generic_hair_bone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "generic_hair.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_generic_hair_bone_complex_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("hair",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 2)
        self.assertEqual({element["name"] for element in data["elements"]}, {"hair_front_1_cube", "hair_front_2_cube"})
        self.assertEqual(report["complex_split"][0]["bone_name"], "hair_front")

    def test_hybrid_mode_uses_special_cubes_for_hair_and_cuboids_for_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hybrid_body_hair.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_hybrid_body_hair_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                mode="hybrid",
                target_height=4.0,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.mode, "hybrid")
        self.assertTrue(result.hybrid.enabled)
        self.assertEqual(result.hybrid.mesh_strategy, "special_cubes")
        self.assertIn("hair", result.hybrid.special_cube_bones)
        self.assertEqual(
            {element["name"] for element in data["elements"]},
            {"body_cube", "hair_front_1_cube", "hair_front_2_cube"},
        )
        self.assertEqual(report["mode"], "hybrid")
        self.assertEqual(report["hybrid"]["mesh_strategy"], "special_cubes")
        self.assertIn("hair", report["hybrid"]["special_cube_bones"])
        self.assertEqual(report["complex_split"][0]["bone_name"], "hair_front")
        self.assertEqual(
            {subpart["name"] for subpart in report["complex_split"][0]["subparts"]},
            {"hair_front_1", "hair_front_2"},
        )

    def test_hybrid_mode_auto_splits_oversized_generic_part(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "oversized_back_panel.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_oversized_back_panel_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                mode="hybrid",
                target_height=4.0,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertGreater(len(result.cubes), 1)
        self.assertEqual(report["complex_split"][0]["bone_name"], "back_panel")
        self.assertEqual({subpart["method"] for subpart in report["complex_split"][0]["subparts"]}, {"auto_spatial_grid"})
        self.assertLess(max(element["to"][0] - element["from"][0] for element in data["elements"]), 16.0)

    def test_hybrid_mode_splits_regular_detail_bone_by_material(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "material_detail_foot.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_material_detail_foot_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                mode="hybrid",
                target_height=10.0,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        self.assertIn("body_cube", element_names)
        self.assertIn("foot_skin_cube", element_names)
        self.assertIn("foot_shoe_cube", element_names)
        self.assertIn("foot_heel_cube", element_names)
        self.assertGreater(len(result.cubes), 2)

        foot_split = next(item for item in report["complex_split"] if item["bone_name"] == "foot")
        self.assertEqual(
            {subpart["name"] for subpart in foot_split["subparts"]},
            {"foot_skin", "foot_shoe", "foot_heel"},
        )
        self.assertEqual({subpart["method"] for subpart in foot_split["subparts"]}, {"regular_material_component"})
        self.assertEqual(report["hybrid_detail_split"]["min_faces"], 200)
        self.assertTrue(report["hybrid_detail_split"]["by_material"])
        self.assertTrue(report["hybrid_detail_split"]["by_connected_component"])

    def test_hybrid_detail_split_config_can_disable_regular_detail_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "material_detail_foot.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_material_detail_foot_fixture(model_path)
            config_path.write_text(
                json.dumps(
                    {
                        "hybrid_detail_split": {
                            "by_material": False,
                            "by_connected_component": False,
                            "min_faces": 300,
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                mode="hybrid",
                target_height=10.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        self.assertIn("foot_cube", element_names)
        self.assertNotIn("foot_skin_cube", element_names)
        self.assertEqual(len(result.cubes), 2)
        self.assertEqual(report["complex_split"], [])
        self.assertEqual(report["hybrid_detail_split"]["min_faces"], 300)
        self.assertFalse(report["hybrid_detail_split"]["by_material"])
        self.assertFalse(report["hybrid_detail_split"]["by_connected_component"])

    def test_hybrid_detail_split_spatially_splits_compact_single_material_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "spatial_detail.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_spatial_detail_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                mode="hybrid",
                target_height=12.0,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        detail_names = {name for name in element_names if name.startswith("detail_part_")}
        self.assertGreaterEqual(len(detail_names), 2)
        self.assertNotIn("detail_part_cube", element_names)
        self.assertGreater(len(result.cubes), 2)

        detail_split = next(item for item in report["complex_split"] if item["bone_name"] == "detail_part")
        self.assertEqual({subpart["method"] for subpart in detail_split["subparts"]}, {"regular_spatial_detail"})

    def test_head_hair_continuity_merges_tiny_middle_bucket_and_expands_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hair_continuity.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_hair_continuity_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 2)
        self.assertEqual({element["name"] for element in data["elements"]}, {"hair_front_left_cube", "hair_front_right_cube"})

        head_split = report["complex_split"][0]
        self.assertEqual(head_split["bone_name"], "head")
        self.assertEqual(head_split["merged_tiny_hair_buckets"], 1)
        self.assertEqual(head_split["expanded_hair_bucket_overlap"], 2)
        self.assertEqual(
            {subpart["name"] for subpart in head_split["subparts"]},
            {"hair_front_left", "hair_front_right"},
        )

    def test_head_hair_split_uses_depth_buckets_for_deep_hair_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "deep_head_hair.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_deep_head_hair_fixture(model_path)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        element_names = {element["name"] for element in data["elements"]}
        self.assertTrue(any(name.startswith("hair_back") and "front_cube" in name for name in element_names))
        self.assertTrue(any(name.startswith("hair_back") and "back_cube" in name for name in element_names))

        subparts = {subpart["name"] for subpart in report["complex_split"][0]["subparts"]}
        self.assertTrue(any(name.startswith("hair_back") and "front" in name for name in subparts))
        self.assertTrue(any(name.startswith("hair_back") and "back" in name for name in subparts))

    def test_large_head_hair_bucket_refines_into_smaller_spatial_parts(self) -> None:
        faces: list[SplitFace] = []
        vertex_index = 0
        for x_index in range(5):
            for z_index in range(5):
                center_x = -1.6 + x_index * 0.8
                center_z = -1.6 + z_index * 0.8
                center_y = 1.4 + ((x_index + z_index) % 3) * 0.05
                for _repeat in range(100):
                    vertex_keys = [("rounded_hair", vertex_index + offset) for offset in range(3)]
                    vertex_index += 3
                    faces.append(
                        SplitFace(
                            owner_bone=1,
                            bone_name="head",
                            points=[
                                [center_x - 0.08, center_y, center_z],
                                [center_x + 0.08, center_y, center_z],
                                [center_x, center_y + 0.08, center_z + 0.08],
                            ],
                            vertex_keys=vertex_keys,
                            material_name="back hair",
                        )
                    )

        result = split_hair_part_faces(
            "hair_back_mid_left_back_upper",
            faces,
            ([-2.0, 0.0, -2.0], [2.0, 10.0, 2.0]),
        )

        self.assertGreaterEqual(len(result.parts), 4)
        self.assertTrue(all(len(part.faces) < len(faces) for part in result.parts))
        self.assertTrue(any(part.name.endswith("_left_front") for part in result.parts))
        self.assertTrue(any(part.name.endswith("_right_back") for part in result.parts))

    def test_complex_split_keeps_face_feature_cubes_on_source_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "face_features.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            write_head_face_feature_fixture(model_path)

            convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        elements = {element["name"]: element for element in data["elements"]}
        self.assertEqual(
            set(elements),
            {
                "brow_l_cube",
                "brow_r_cube",
                "eye_l_cube",
                "eye_r_cube",
                "head_core_cube",
                "mouth_cube",
                "nose_cube",
            },
        )
        head_min_z = elements["head_core_cube"]["from"][2]
        head_max_z = elements["head_core_cube"]["to"][2]
        for feature_name in ("brow_l_cube", "brow_r_cube", "eye_l_cube", "eye_r_cube", "mouth_cube", "nose_cube"):
            self.assertEqual(elements[feature_name]["from"][2], elements[feature_name]["to"][2])
            self.assertGreaterEqual(elements[feature_name]["from"][2], head_min_z)
            self.assertLessEqual(elements[feature_name]["to"][2], head_max_z)

    def test_complex_split_head_uses_vrm1_humanoid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "vroid_style.vrm"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_vrm1_head_complex_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                complex_split=("head",),
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 3)
        self.assertEqual({element["name"] for element in data["elements"]}, {"head_core_cube", "hair_front_cube", "hair_back_cube"})
        self.assertEqual(report["complex_split"][0]["bone_name"], "J_Upper_05")

    def test_oriented_cubes_config_rotates_complex_head_subparts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "oriented_head.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_rotated_head_complex_fixture(model_path)
            config_path.write_text(
                json.dumps(
                    {
                        "complex_split": {"enabled": True, "bones": ["head"]},
                        "oriented_cubes": {"enabled": True, "bones": ["head"]},
                    }
                ),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.oriented_cubes), 3)
        self.assertEqual(report["totals"]["oriented_cubes"], 3)
        self.assertEqual({item["source"] for item in report["oriented_cubes"]}, {"bone_world_matrix"})
        for element in data["elements"]:
            self.assertEqual(element["rotation"], [0.0, 0.0, 90.0])

    def test_oriented_cubes_config_rotates_selected_bone_cube_from_child_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "slanted_thumb.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_slanted_thumb_fixture(model_path)
            config_path.write_text(
                json.dumps({"oriented_cubes": {"enabled": True, "bones": ["Thumb"], "scope": "bone_cubes"}}),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.oriented_cubes), 1)
        self.assertEqual(report["oriented_cubes"][0]["source"], "bone_direction")
        self.assertEqual(data["elements"][0]["name"], "J_Bip_L_Thumb1_cube")
        self.assertAlmostEqual(data["elements"][0]["rotation"][2], -45.0)

    def test_hybrid_mode_auto_rotates_long_bone_cube_from_child_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "slanted_thumb.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            write_slanted_thumb_fixture(model_path)

            result = convert_model(
                model_path,
                output_path,
                mode="hybrid",
                target_height=4.0,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.oriented_cubes), 1)
        self.assertEqual(report["oriented_cubes"][0]["source"], "auto_bone_direction")
        self.assertEqual(data["elements"][0]["name"], "J_Bip_L_Thumb1_cube")
        self.assertAlmostEqual(data["elements"][0]["rotation"][2], -45.0)

    def test_cleanup_config_deletes_small_parts_before_writing_bbmodel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cleanup_delete.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_cleanup_small_part_fixture(model_path)
            config_path.write_text(
                json.dumps({"cleanup": {"delete_small_parts": True, "min_faces": 2}}),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 1)
        self.assertEqual([element["name"] for element in data["elements"]], ["root_joint_cube"])
        self.assertEqual(report["totals"]["deleted_small_parts"], 1)
        self.assertEqual(report["cleanup"]["deleted_parts"][0]["name"], "child_joint")
        self.assertEqual(report["cleanup"]["deleted_parts"][0]["reason"], "faces<2")

    def test_cleanup_deletes_small_connected_component_inside_bone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cleanup_component_delete.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_cleanup_disconnected_component_fixture(model_path)
            config_path.write_text(
                json.dumps({"cleanup": {"delete_small_parts": True, "min_faces": 2}}),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 1)
        self.assertEqual(report["totals"]["deleted_small_parts"], 1)
        self.assertEqual(data["elements"][0]["name"], "root_joint_cube")
        self.assertEqual(data["elements"][0]["from"], [-2.0, 0.0, 0.0])
        self.assertEqual(data["elements"][0]["to"], [2.0, 4.0, 0.0])

        deleted = report["cleanup"]["deleted_parts"][0]
        self.assertEqual(deleted["name"], "root_joint_component_2")
        self.assertEqual(deleted["owner_bone_name"], "root_joint")
        self.assertEqual(deleted["reason"], "faces<2")
        self.assertEqual(deleted["faces"], 1)

    def test_cleanup_config_merges_small_parts_to_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cleanup_merge.gltf"
            output_path = Path(temp_dir) / "model.bbmodel"
            report_path = Path(temp_dir) / "report.json"
            config_path = Path(temp_dir) / "config.json"
            write_cleanup_small_part_fixture(model_path)
            config_path.write_text(
                json.dumps({"cleanup": {"merge_small_parts_to_parent": True, "min_faces": 2}}),
                encoding="utf-8",
            )

            result = convert_model(
                model_path,
                output_path,
                target_height=4.0,
                config_path=config_path,
                report_path=report_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result.cubes), 1)
        self.assertEqual([element["name"] for element in data["elements"]], ["root_joint_cube"])
        self.assertEqual(result.cubes[0].faces, 3)
        self.assertEqual(report["totals"]["merged_small_parts"], 1)
        merged = report["cleanup"]["merged_parts"][0]
        self.assertEqual(merged["name"], "child_joint")
        self.assertEqual(merged["target_bone_name"], "root_joint")


def write_two_bone_cuboid_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 2.0, 0.0),
        ]
    )
    joints = bytes(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
    )
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(6))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb convert test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "child_joint", "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "meshes": [
            {
                "name": "two_triangles",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 6, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 6, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_thin_nonzero_triangle_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (0.0, 0.0, 0.0),
            (0.001, 0.0, 0.001),
            (0.0, 1.0, 0.0),
        ]
    )
    joints = bytes([0, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(3))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb thin nonzero convert test"},
        "scene": 0,
        "scenes": [{"nodes": [1]}],
        "nodes": [
            {"name": "root_joint"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0]}],
        "meshes": [
            {
                "name": "thin_nonzero_triangle",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_quality_diagnostics_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (0.0, 0.0, 0.0),
            (10.0, 1.0, 0.0),
            (0.2, 0.0, 0.0),
        ]
    )
    joints = bytes([0, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(3))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb quality diagnostics convert test"},
        "scene": 0,
        "scenes": [{"nodes": [1, 2]}],
        "nodes": [
            {"name": "root_joint"},
            {"name": "skinned_mesh_node", "mesh": 0, "skin": 0},
            {"name": "static_mesh_node", "mesh": 0},
        ],
        "skins": [{"joints": [0]}],
        "meshes": [
            {
                "name": "quality_diagnostics_mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_unskinned_parent_fixture(path: Path, ignored_static_material: bool = False) -> None:
    skinned_positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    joints = bytes([0, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(3))
    static_positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    payload = base64.b64encode(skinned_positions + joints + weights + static_positions).decode("ascii")
    static_offset = len(skinned_positions) + len(joints) + len(weights)
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb unskinned parent convert test"},
        "scene": 0,
        "scenes": [{"nodes": [0, 2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "child_joint", "translation": [0.0, 1.0, 0.0], "children": [3]},
            {"name": "skinned_mesh_node", "mesh": 0, "skin": 0},
            {"name": "static_mesh_node", "mesh": 1},
        ],
        "skins": [{"joints": [0, 1]}],
        "meshes": [
            {
                "name": "skinned_triangle",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            },
            {
                "name": "static_triangle",
                "primitives": [
                    {
                        "attributes": {"POSITION": 3},
                        **({"material": 0} if ignored_static_material else {}),
                        "mode": 4,
                    }
                ],
            },
        ],
        **({"materials": [{"name": "mmd_tools_rigid_0"}]} if ignored_static_material else {}),
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(skinned_positions)},
            {"buffer": 0, "byteOffset": len(skinned_positions), "byteLength": len(joints)},
            {
                "buffer": 0,
                "byteOffset": len(skinned_positions) + len(joints),
                "byteLength": len(weights),
            },
            {"buffer": 0, "byteOffset": static_offset, "byteLength": len(static_positions)},
        ],
        "buffers": [
            {
                "byteLength": len(skinned_positions) + len(joints) + len(weights) + len(static_positions),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_near_planar_triangle_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.001),
            (-1.0, 1.0, 0.0),
        ]
    )
    joints = bytes([0, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(3))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb near planar convert test"},
        "scene": 0,
        "scenes": [{"nodes": [1]}],
        "nodes": [
            {"name": "root_joint"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0]}],
        "meshes": [
            {
                "name": "near_planar_triangle",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_helper_merge_cuboid_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    joints = bytes([1, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(3))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb helper merge convert test"},
        "scene": 0,
        "scenes": [{"nodes": [3]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "twist_helper_joint", "children": [2], "translation": [0.0, 1.0, 0.0]},
            {"name": "child_joint", "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1, 2]}],
        "meshes": [
            {
                "name": "helper_triangle",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_cleanup_small_part_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.5, 1.5, 0.0),
        ]
    )
    joints = bytes([0, 0, 0, 0] * 6 + [1, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(9))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb cleanup convert test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "child_joint", "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "meshes": [
            {
                "name": "cleanup_parts",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 9, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 9, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 9, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_cleanup_disconnected_component_fixture(path: Path) -> None:
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (10.0, 10.0, 0.0),
        (10.2, 10.0, 0.0),
        (10.0, 10.2, 0.0),
    ]
    positions = b"".join(struct.pack("<fff", *point) for point in points)
    joints = bytes([0, 0, 0, 0] * len(points))
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb cleanup component convert test"},
        "scene": 0,
        "scenes": [{"nodes": [1]}],
        "nodes": [
            {"name": "root_joint"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0]}],
        "meshes": [
            {
                "name": "cleanup_component_parts",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": len(points), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": len(points), "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_slanted_thumb_fixture(path: Path) -> None:
    positions = b"".join(
        struct.pack("<fff", *point)
        for point in [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.2, 0.0, 0.0),
        ]
    )
    joints = bytes([1, 0, 0, 0] * 3)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(3))
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb slanted thumb test"},
        "scene": 0,
        "scenes": [{"nodes": [3]}],
        "nodes": [
            {"name": "Root", "children": [1]},
            {"name": "J_Bip_L_Thumb1", "children": [2]},
            {"name": "J_Bip_L_Thumb2", "translation": [1.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1, 2]}],
        "meshes": [
            {
                "name": "slanted_thumb",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_head_complex_fixture(path: Path) -> None:
    model = build_head_complex_model("head", include_vrm_metadata=False)
    path.write_text(json.dumps(model), encoding="utf-8")


def write_large_head_core_fixture(path: Path) -> None:
    def repeated_triangles(points: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
        result: list[tuple[float, float, float]] = []
        for _ in range(count):
            result.extend(points)
        return result

    model = build_head_feature_model(
        [
            (repeated_triangles([(-1.2, 0.1, -1.1), (-0.8, 0.1, -1.1), (-1.0, 0.5, -1.1)], 225), 0),
            (repeated_triangles([(-1.1, 0.6, -1.1), (-0.7, 0.6, -1.1), (-0.9, 1.0, -1.1)], 225), 0),
            (repeated_triangles([(0.7, 0.1, 1.1), (1.1, 0.1, 1.1), (0.9, 0.5, 1.1)], 225), 0),
            (repeated_triangles([(0.8, 0.6, 1.1), (1.2, 0.6, 1.1), (1.0, 1.0, 1.1)], 225), 0),
        ],
        [{"name": "face skin"}],
    )
    path.write_text(json.dumps(model), encoding="utf-8")


def write_large_head_core_with_face_features_fixture(path: Path, include_front_hair: bool = False) -> None:
    def repeated_triangle(points: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
        result: list[tuple[float, float, float]] = []
        for _ in range(count):
            result.extend(points)
        return result

    specs = [
        (repeated_triangle([(-1.2, 1.0, -1.0), (1.2, 1.0, -1.0), (-1.2, 1.3, -1.0)], 180), 0),
        (repeated_triangle([(-1.2, 1.8, -1.0), (1.2, 1.8, -1.0), (-1.2, 2.1, -1.0)], 180), 0),
        (repeated_triangle([(-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.25, 1.0)], 180), 0),
        (repeated_triangle([(-1.0, 1.45, 1.0), (1.0, 1.45, 1.0), (0.0, 1.75, 1.0)], 180), 0),
        (repeated_triangle([(-1.0, 1.9, 1.0), (1.0, 1.9, 1.0), (0.0, 2.15, 1.0)], 180), 0),
        ([(-0.55, 1.58, 1.05), (-0.3, 1.58, 1.05), (-0.425, 1.68, 1.05)], 1),
        ([(0.3, 1.58, 1.05), (0.55, 1.58, 1.05), (0.425, 1.68, 1.05)], 1),
        ([(-0.2, 1.32, 1.05), (0.2, 1.32, 1.05), (0.0, 1.38, 1.05)], 2),
        ([(-0.6, 1.8, 1.05), (-0.25, 1.8, 1.05), (-0.425, 1.88, 1.05)], 3),
        ([(0.25, 1.8, 1.05), (0.6, 1.8, 1.05), (0.425, 1.88, 1.05)], 3),
    ]
    materials = [
        {"name": "face_skin"},
        {"name": "eye material"},
        {"name": "mouth material"},
        {"name": "eyebrow material"},
    ]
    if include_front_hair:
        specs.append((repeated_triangle([(-1.4, 1.1, 1.2), (1.4, 1.1, 1.2), (0.0, 1.6, 1.2)], 260), 4))
        specs.append((repeated_triangle([(-1.4, 1.1, 1.25), (1.4, 2.45, 1.25), (1.2, 2.45, 1.25)], 260), 4))
        materials.append({"name": "front hair"})

    model = build_head_feature_model(specs, materials)
    path.write_text(json.dumps(model), encoding="utf-8")


def write_explicit_eye_head_fixture(
    path: Path,
    include_front_hair: bool = False,
    include_back_hair: bool = False,
) -> None:
    def repeated_triangle(points: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
        result: list[tuple[float, float, float]] = []
        for _ in range(count):
            result.extend(points)
        return result

    chunks: list[bytes] = []
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, int | str]] = []
    primitives: list[dict[str, object]] = []
    byte_offset = 0

    def append_chunk(data: bytes) -> int:
        nonlocal byte_offset
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)})
        chunks.append(data)
        byte_offset += len(data)
        return view_index

    def add_primitive(points: list[tuple[float, float, float]], joint: int, material: int) -> None:
        positions = b"".join(struct.pack("<fff", *point) for point in points)
        joints = bytes([joint, 0, 0, 0] * len(points))
        weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
        position_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(positions), "componentType": 5126, "count": len(points), "type": "VEC3"})
        joints_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(joints), "componentType": 5121, "count": len(points), "type": "VEC4"})
        weights_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(weights), "componentType": 5126, "count": len(points), "type": "VEC4"})
        primitives.append(
            {
                "attributes": {"POSITION": position_accessor, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor},
                "material": material,
                "mode": 4,
            }
        )

    add_primitive(repeated_triangle([(-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.3, -1.0)], 180), 1, 0)
    add_primitive(repeated_triangle([(-1.0, 1.9, -1.0), (1.0, 1.9, -1.0), (-1.0, 2.2, -1.0)], 180), 1, 0)
    add_primitive(repeated_triangle([(-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.25, 1.0)], 180), 1, 0)
    add_primitive(repeated_triangle([(-1.0, 1.45, 1.0), (1.0, 1.45, 1.0), (0.0, 1.75, 1.0)], 180), 1, 0)
    add_primitive(repeated_triangle([(-1.0, 1.9, 1.0), (1.0, 1.9, 1.0), (0.0, 2.15, 1.0)], 180), 1, 0)
    add_primitive([(-0.55, 1.58, 1.05), (-0.3, 1.58, 1.05), (-0.425, 1.68, 1.05)], 1, 1)
    add_primitive([(0.3, 1.58, 1.05), (0.55, 1.58, 1.05), (0.425, 1.68, 1.05)], 1, 1)
    add_primitive([(-0.2, 1.32, 1.05), (0.2, 1.32, 1.05), (0.0, 1.38, 1.05)], 1, 2)
    add_primitive([(-0.55, 1.58, 1.2), (-0.3, 1.58, 1.2), (-0.425, 1.68, 1.2)], 2, 1)
    add_primitive([(0.3, 1.58, 1.2), (0.55, 1.58, 1.2), (0.425, 1.68, 1.2)], 3, 1)
    if include_front_hair:
        add_primitive(repeated_triangle([(-1.2, 1.2, 1.25), (1.2, 1.2, 1.25), (0.0, 2.45, 1.25)], 260), 1, 3)
    if include_back_hair:
        back_hair_material = 4 if include_front_hair else 3
        add_primitive(
            repeated_triangle([(-1.2, 1.2, 1.25), (1.2, 1.2, 1.25), (0.0, 2.45, 1.25)], 260),
            1,
            back_hair_material,
        )

    payload_bytes = b"".join(chunks)
    payload = base64.b64encode(payload_bytes).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb explicit eye head test"},
        "scene": 0,
        "scenes": [{"nodes": [4]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "head", "translation": [0.0, 1.0, 0.0], "children": [2, 3]},
            {"name": "目.R"},
            {"name": "目.L"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1, 2, 3]}],
        "materials": [
            {"name": "face_skin"},
            {"name": "eye material"},
            {"name": "mouth material"},
            *([{"name": "front hair"}] if include_front_hair else []),
            *([{"name": "back hair"}] if include_back_hair else []),
        ],
        "meshes": [{"name": "explicit_eye_head", "primitives": primitives}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(payload_bytes), "uri": f"data:application/octet-stream;base64,{payload}"}],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_rotated_head_complex_fixture(path: Path) -> None:
    model = build_head_complex_model(
        "head",
        include_vrm_metadata=False,
        head_rotation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
    )
    path.write_text(json.dumps(model), encoding="utf-8")


def write_vrm1_head_complex_fixture(path: Path) -> None:
    model = build_head_complex_model("J_Upper_05", include_vrm_metadata=True)
    write_json_glb(path, model)


def write_head_tiny_component_fixture(path: Path) -> None:
    points = [
        (-0.5, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (-0.5, 2.0, 0.0),
        (0.5, 2.0, 0.0),
        (-0.1, 1.4, -1.0),
        (0.1, 1.4, -1.0),
        (0.0, 1.6, -1.0),
    ]
    indices = [0, 1, 2, 1, 3, 2, 4, 5, 6]
    positions = b"".join(struct.pack("<fff", *point) for point in points)
    joints = bytes([1, 0, 0, 0] * len(points))
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
    index_bytes = b"".join(struct.pack("<H", index) for index in indices)
    payload = base64.b64encode(positions + joints + weights + index_bytes).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb tiny component split test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "head", "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "meshes": [
            {
                "name": "head_tiny_component",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "indices": 3,
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": len(points), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": len(points), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
            {
                "buffer": 0,
                "byteOffset": len(positions) + len(joints) + len(weights),
                "byteLength": len(index_bytes),
            },
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights) + len(index_bytes),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_head_multiple_hair_components_fixture(path: Path) -> None:
    model = build_head_feature_model(
        [
            ([(-0.4, 1.0, 0.0), (0.4, 1.0, 0.0), (0.0, 1.8, 0.0)], 0),
            ([(-0.5, 1.6, -1.0), (-0.2, 1.6, -1.0), (-0.35, 2.1, -1.0)], 1),
            ([(0.2, 1.6, -1.0), (0.5, 1.6, -1.0), (0.35, 2.1, -1.0)], 1),
        ],
        [{"name": "face_skin"}, {"name": "front hair"}],
    )
    path.write_text(json.dumps(model), encoding="utf-8")


def write_generic_hair_bone_complex_fixture(path: Path) -> None:
    model = build_head_feature_model(
        [
            (
                [
                    (-0.5, 1.6, -1.0),
                    (-0.2, 1.6, -1.0),
                    (-0.35, 2.1, -1.0),
                    (0.2, 1.6, -1.0),
                    (0.5, 1.6, -1.0),
                    (0.35, 2.1, -1.0),
                ],
                0,
            ),
        ],
        [{"name": "front hair"}],
        bone_name="hair_front",
    )
    path.write_text(json.dumps(model), encoding="utf-8")


def write_hybrid_body_hair_fixture(path: Path) -> None:
    points = [
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (-0.5, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.5, 1.0, 0.0),
        (-0.5, 1.0, 0.0),
        (-0.4, 1.3, -0.8),
        (-0.1, 1.3, -0.8),
        (-0.25, 2.0, -0.8),
        (0.1, 1.3, -0.8),
        (0.4, 1.3, -0.8),
        (0.25, 2.0, -0.8),
    ]
    positions = b"".join(struct.pack("<fff", *point) for point in points)
    joints = bytes([1, 0, 0, 0] * 6 + [2, 0, 0, 0] * 6)
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb hybrid mode test"},
        "scene": 0,
        "scenes": [{"nodes": [3]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "body", "children": [2], "translation": [0.0, 0.0, 0.0]},
            {"name": "hair_front", "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1, 2]}],
        "meshes": [
            {
                "name": "hybrid_body_hair",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": len(points), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": len(points), "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_hair_continuity_fixture(path: Path) -> None:
    model = build_head_feature_model(
        [
            ([(-2.4, 1.0, -1.4), (-2.0, 1.0, -1.4), (-2.2, 1.6, -1.4)], 0),
            ([(-2.5, 1.1, -1.3), (-1.9, 1.0, -1.3), (-2.2, 1.7, -1.3)], 0),
            ([(-2.3, 0.9, -1.2), (-1.8, 1.0, -1.2), (-2.0, 1.7, -1.2)], 0),
            ([(-2.1, 1.1, -1.1), (-1.7, 1.1, -1.1), (-1.9, 1.8, -1.1)], 0),
            ([(-0.1, 1.0, -1.2), (0.1, 1.0, -1.2), (0.0, 1.6, -1.2)], 0),
            ([(1.7, 1.0, -1.4), (2.1, 1.0, -1.4), (1.9, 1.7, -1.4)], 0),
            ([(1.8, 1.1, -1.3), (2.4, 1.0, -1.3), (2.1, 1.8, -1.3)], 0),
            ([(1.9, 0.9, -1.2), (2.5, 1.0, -1.2), (2.2, 1.7, -1.2)], 0),
            ([(1.7, 1.1, -1.1), (2.1, 1.1, -1.1), (1.9, 1.8, -1.1)], 0),
        ],
        [{"name": "front hair"}],
    )
    path.write_text(json.dumps(model), encoding="utf-8")


def write_deep_head_hair_fixture(path: Path) -> None:
    points = [
        (-0.5, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (-0.5, 2.0, 0.0),
        (-0.3, 1.2, -1.0),
        (0.3, 1.2, -1.0),
        (-0.3, 2.1, -1.0),
        (0.3, 2.1, -1.0),
        (-0.3, 1.2, 1.0),
        (0.3, 1.2, 1.0),
        (-0.3, 2.1, 1.0),
        (0.3, 2.1, 1.0),
    ]
    front_faces = [3, 4, 5, 4, 6, 5]
    back_faces = [7, 9, 8, 8, 9, 10]
    side_faces = [3, 5, 7, 7, 5, 9, 4, 8, 6, 8, 10, 6]
    indices = [0, 1, 2] + (front_faces + back_faces + side_faces) * 4
    positions = b"".join(struct.pack("<fff", *point) for point in points)
    joints = bytes([1, 0, 0, 0] * len(points))
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
    index_bytes = b"".join(struct.pack("<H", index) for index in indices)
    payload = base64.b64encode(positions + joints + weights + index_bytes).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb deep hair split test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "head", "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "materials": [{"name": "face_skin"}, {"name": "back hair"}],
        "meshes": [
            {
                "name": "deep_head_hair",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    },
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "indices": 4,
                        "material": 1,
                        "mode": 4,
                    },
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": len(points), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": len(points), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5123, "count": len(indices) - 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
            {
                "buffer": 0,
                "byteOffset": len(positions) + len(joints) + len(weights),
                "byteLength": 6,
            },
            {
                "buffer": 0,
                "byteOffset": len(positions) + len(joints) + len(weights) + 6,
                "byteLength": len(index_bytes) - 6,
            },
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights) + len(index_bytes),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_oversized_back_panel_fixture(path: Path) -> None:
    points: list[tuple[float, float, float]] = []
    x_segments = 8
    y_segments = 4
    for y_index in range(y_segments):
        y0 = y_index * 0.5
        y1 = (y_index + 1) * 0.5
        for x_index in range(x_segments):
            x0 = -4.0 + x_index
            x1 = x0 + 1.0
            points.extend(
                [
                    (x0, y0, 0.0),
                    (x1, y0, 0.0),
                    (x0, y1, 0.0),
                    (x1, y0, 0.0),
                    (x1, y1, 0.0),
                    (x0, y1, 0.0),
                ]
            )

    positions = b"".join(struct.pack("<fff", *point) for point in points)
    joints = bytes([0, 0, 0, 0] * len(points))
    weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
    payload = base64.b64encode(positions + joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb oversized generic split test"},
        "scene": 0,
        "scenes": [{"nodes": [1]}],
        "nodes": [
            {"name": "back_panel"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0]}],
        "meshes": [
            {
                "name": "oversized_back_panel",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": len(points), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": len(points), "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(positions) + len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_material_detail_foot_fixture(path: Path) -> None:
    chunks: list[bytes] = []
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, int | str]] = []
    primitives: list[dict[str, object]] = []
    byte_offset = 0

    def append_chunk(data: bytes) -> int:
        nonlocal byte_offset
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)})
        chunks.append(data)
        byte_offset += len(data)
        return view_index

    def add_primitive(points: list[tuple[float, float, float]], joint: int, material: int) -> None:
        positions = b"".join(struct.pack("<fff", *point) for point in points)
        joints = bytes([joint, 0, 0, 0] * len(points))
        weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
        position_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(positions), "componentType": 5126, "count": len(points), "type": "VEC3"})
        joints_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(joints), "componentType": 5121, "count": len(points), "type": "VEC4"})
        weights_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(weights), "componentType": 5126, "count": len(points), "type": "VEC4"})
        primitives.append(
            {
                "attributes": {"POSITION": position_accessor, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor},
                "material": material,
                "mode": 4,
            }
        )

    def repeated_triangle(points: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
        result: list[tuple[float, float, float]] = []
        for _ in range(count):
            result.extend(points)
        return result

    add_primitive(repeated_triangle([(-0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 10.0, 0.0)], 2), 0, 0)
    add_primitive(repeated_triangle([(0.0, 0.6, -0.2), (0.8, 0.6, -0.2), (0.2, 1.2, -0.1)], 75), 1, 1)
    add_primitive(repeated_triangle([(0.0, 0.0, 0.0), (0.9, 0.0, 0.0), (0.3, 0.5, 0.8)], 75), 1, 2)
    add_primitive(repeated_triangle([(0.2, 0.0, -0.7), (0.7, 0.0, -0.7), (0.45, 1.0, -0.7)], 75), 1, 3)

    payload_bytes = b"".join(chunks)
    payload = base64.b64encode(payload_bytes).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb material detail split test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "body", "children": [1]},
            {"name": "foot"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "materials": [{"name": "body"}, {"name": "skin"}, {"name": "shoe"}, {"name": "heel"}],
        "meshes": [{"name": "material_detail_foot", "primitives": primitives}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(payload_bytes), "uri": f"data:application/octet-stream;base64,{payload}"}],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_spatial_detail_fixture(path: Path) -> None:
    chunks: list[bytes] = []
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, int | str]] = []
    primitives: list[dict[str, object]] = []
    byte_offset = 0

    def append_chunk(data: bytes) -> int:
        nonlocal byte_offset
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)})
        chunks.append(data)
        byte_offset += len(data)
        return view_index

    def repeated_triangle(points: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
        result: list[tuple[float, float, float]] = []
        for _ in range(count):
            result.extend(points)
        return result

    def add_primitive(points: list[tuple[float, float, float]], joint: int, material: int) -> None:
        positions = b"".join(struct.pack("<fff", *point) for point in points)
        joints = bytes([joint, 0, 0, 0] * len(points))
        weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
        position_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(positions), "componentType": 5126, "count": len(points), "type": "VEC3"})
        joints_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(joints), "componentType": 5121, "count": len(points), "type": "VEC4"})
        weights_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(weights), "componentType": 5126, "count": len(points), "type": "VEC4"})
        primitives.append(
            {
                "attributes": {"POSITION": position_accessor, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor},
                "material": material,
                "mode": 4,
            }
        )

    add_primitive(repeated_triangle([(-0.3, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 12.0, 0.0)], 2), 0, 0)
    add_primitive(repeated_triangle([(-0.9, 10.2, 0.8), (-0.25, 10.2, 1.4), (-0.55, 11.0, 1.0)], 120), 1, 1)
    add_primitive(repeated_triangle([(0.25, 10.2, 1.4), (0.9, 10.2, 0.8), (0.55, 11.0, 1.0)], 120), 1, 1)

    payload_bytes = b"".join(chunks)
    payload = base64.b64encode(payload_bytes).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb spatial detail split test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "body", "children": [1]},
            {"name": "detail_part"},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "materials": [{"name": "body"}, {"name": "opaque_detail"}],
        "meshes": [{"name": "spatial_detail", "primitives": primitives}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(payload_bytes), "uri": f"data:application/octet-stream;base64,{payload}"}],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_head_face_feature_fixture(path: Path) -> None:
    model = build_head_feature_model(
        [
            ([(-0.6, 1.0, -0.5), (0.6, 1.0, 0.5), (-0.6, 2.0, -0.5), (0.6, 1.0, 0.5), (0.6, 2.0, 0.5), (-0.6, 2.0, -0.5)], 0),
            ([(-0.35, 1.65, -0.2), (-0.2, 1.65, -0.2), (-0.275, 1.75, -0.2)], 1),
            ([(0.2, 1.65, -0.2), (0.35, 1.65, -0.2), (0.275, 1.75, -0.2)], 1),
            ([(-0.15, 1.35, -0.2), (0.15, 1.35, -0.2), (0.0, 1.42, -0.2)], 2),
            ([(-0.38, 1.82, -0.2), (-0.18, 1.82, -0.2), (-0.28, 1.88, -0.2)], 3),
            ([(0.18, 1.82, -0.2), (0.38, 1.82, -0.2), (0.28, 1.88, -0.2)], 3),
            ([(-0.06, 1.55, -0.2), (0.06, 1.55, -0.2), (0.0, 1.66, -0.2)], 4),
        ],
        [
            {"name": "face_skin"},
            {"name": "eye material"},
            {"name": "mouth material"},
            {"name": "eyebrow material"},
            {"name": "nose material"},
        ],
    )
    path.write_text(json.dumps(model), encoding="utf-8")


def build_head_feature_model(
    primitive_specs: list[tuple[list[tuple[float, float, float]], int]],
    materials: list[dict[str, str]],
    bone_name: str = "head",
) -> dict[str, object]:
    chunks: list[bytes] = []
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, int | str]] = []
    primitives: list[dict[str, object]] = []
    byte_offset = 0

    def append_chunk(data: bytes) -> int:
        nonlocal byte_offset
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)})
        chunks.append(data)
        byte_offset += len(data)
        return view_index

    for points, material in primitive_specs:
        positions = b"".join(struct.pack("<fff", *point) for point in points)
        joints = bytes([1, 0, 0, 0] * len(points))
        weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)
        position_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(positions), "componentType": 5126, "count": len(points), "type": "VEC3"})
        joints_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(joints), "componentType": 5121, "count": len(points), "type": "VEC4"})
        weights_accessor = len(accessors)
        accessors.append({"bufferView": append_chunk(weights), "componentType": 5126, "count": len(points), "type": "VEC4"})
        primitives.append(
            {
                "attributes": {"POSITION": position_accessor, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor},
                "material": material,
                "mode": 4,
            }
        )

    payload_bytes = b"".join(chunks)
    payload = base64.b64encode(payload_bytes).decode("ascii")
    return {
        "asset": {"version": "2.0", "generator": "gltf2bb head feature split test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": bone_name, "translation": [0.0, 1.0, 0.0]},
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "materials": materials,
        "meshes": [{"name": "head_features", "primitives": primitives}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(payload_bytes), "uri": f"data:application/octet-stream;base64,{payload}"}],
    }


def build_head_complex_model(
    head_bone_name: str,
    include_vrm_metadata: bool,
    head_rotation: list[float] | None = None,
) -> dict[str, object]:
    chunks: list[bytes] = []
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, int | str]] = []
    primitives: list[dict[str, object]] = []
    byte_offset = 0

    def append_chunk(data: bytes) -> int:
        nonlocal byte_offset
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)})
        chunks.append(data)
        byte_offset += len(data)
        return view_index

    def add_primitive(points: list[tuple[float, float, float]], material: int) -> None:
        positions = b"".join(struct.pack("<fff", *point) for point in points)
        joints = bytes([1, 0, 0, 0] * len(points))
        weights = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in points)

        position_accessor = len(accessors)
        accessors.append(
            {"bufferView": append_chunk(positions), "componentType": 5126, "count": len(points), "type": "VEC3"}
        )
        joints_accessor = len(accessors)
        accessors.append(
            {"bufferView": append_chunk(joints), "componentType": 5121, "count": len(points), "type": "VEC4"}
        )
        weights_accessor = len(accessors)
        accessors.append(
            {"bufferView": append_chunk(weights), "componentType": 5126, "count": len(points), "type": "VEC4"}
        )
        primitives.append(
            {
                "attributes": {"POSITION": position_accessor, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor},
                "material": material,
                "mode": 4,
            }
        )

    add_primitive([(-0.4, 1.0, 0.0), (0.4, 1.0, 0.0), (0.0, 1.8, 0.0)], 0)
    add_primitive([(-0.4, 1.8, -1.0), (0.4, 1.8, -1.0), (0.0, 2.2, -1.0)], 1)
    add_primitive([(-0.4, 1.8, 1.0), (0.4, 1.8, 1.0), (0.0, 2.2, 1.0)], 1)

    payload_bytes = b"".join(chunks)
    payload = base64.b64encode(payload_bytes).decode("ascii")
    head_node: dict[str, object] = {"name": head_bone_name, "translation": [0.0, 1.0, 0.0]}
    if head_rotation is not None:
        head_node["rotation"] = head_rotation

    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb complex split convert test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            head_node,
            {"name": "mesh_node", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1]}],
        "materials": [{"name": "face_skin"}, {"name": "long hair"}],
        "meshes": [{"name": "complex_head", "primitives": primitives}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(payload_bytes), "uri": f"data:application/octet-stream;base64,{payload}"}],
    }
    if include_vrm_metadata:
        model["extensionsUsed"] = ["VRMC_vrm"]
        model["extensions"] = {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "humanoid": {"humanBones": {"head": {"node": 1}}},
                "meta": {"name": "VRoid style test model", "version": "1.0"},
            }
        }
    return model


def write_json_glb(path: Path, model: dict[str, object]) -> None:
    json_chunk = json.dumps(model, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    total_length = 12 + 8 + len(json_chunk)
    glb = struct.pack("<4sII", b"glTF", 2, total_length)
    glb += struct.pack("<II", len(json_chunk), 0x4E4F534A)
    glb += json_chunk
    path.write_bytes(glb)


def find_outliner_entry(entries: list[object], uuid: str) -> dict[str, object] | None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("uuid") == uuid:
            return entry
        match = find_outliner_entry(entry.get("children", []), uuid)
        if match is not None:
            return match
    return None


if __name__ == "__main__":
    unittest.main()
