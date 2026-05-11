from __future__ import annotations

import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path

from src.config import preset_config, preset_names
from src.partition import partition_model, partition_report_to_dict


class PartitionModelTest(unittest.TestCase):
    def test_assigns_faces_to_dominant_bones(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "two_bone_faces.gltf"
            write_two_bone_fixture(model_path)

            report = partition_model(model_path)
            data = partition_report_to_dict(report)

        bones = {bone["name"]: bone for bone in data["bones"]}
        self.assertEqual(data["totals"]["faces"], 2)
        self.assertEqual(data["totals"]["assigned_faces"], 2)
        self.assertEqual(data["totals"]["fallback_faces"], 0)
        self.assertEqual(bones["root_joint"]["faces"], 1)
        self.assertEqual(bones["child_joint"]["faces"], 1)

    def test_preset_merges_helper_bone_to_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "helper_bone_faces.gltf"
            write_helper_merge_fixture(model_path)

            report = partition_model(model_path, preset="mmd_humanoid")
            data = partition_report_to_dict(report)

        bones = {bone["name"]: bone for bone in data["bones"]}
        self.assertEqual(data["preset"], "mmd_humanoid")
        self.assertEqual(data["totals"]["original_bones"], 3)
        self.assertEqual(data["totals"]["kept_bones"], 2)
        self.assertEqual(data["totals"]["merged_bones"], 1)
        self.assertNotIn("twist_helper_joint", bones)
        self.assertEqual(bones["root_joint"]["faces"], 1)
        self.assertEqual(bones["child_joint"]["parent"], 0)

        merged = data["bone_resolution"]["merged_to_parent"][0]
        self.assertEqual(merged["name"], "twist_helper_joint")
        self.assertEqual(merged["resolved_to_name"], "root_joint")

    def test_config_can_disable_preset_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "helper_bone_faces.gltf"
            config_path = Path(temp_dir) / "config.json"
            write_helper_merge_fixture(model_path)
            config_path.write_text(json.dumps({"bone_filter": {"enabled": False}}), encoding="utf-8")

            report = partition_model(model_path, preset="mmd_humanoid", config_path=config_path)
            data = partition_report_to_dict(report)

        bones = {bone["name"]: bone for bone in data["bones"]}
        self.assertEqual(data["totals"]["kept_bones"], 3)
        self.assertEqual(data["totals"]["merged_bones"], 0)
        self.assertEqual(bones["twist_helper_joint"]["faces"], 1)

    def test_vroid_preset_keeps_secondary_hair_and_face_eye_bones(self) -> None:
        config = preset_config("vroid")

        self.assertIn("vroid", preset_names())
        self.assertEqual(config.preset, "vroid")
        self.assertIn(r"_end(?:_|$)", config.bone_filter.merge_to_parent_name_regex)
        self.assertIn("hair", config.complex_split.bones)
        self.assertIn("hood", config.complex_split.bones)
        self.assertIn("string", config.complex_split.bones)
        self.assertIn("chest", config.complex_split.bones)
        self.assertIn("shoulder", config.complex_split.bones)
        self.assertIn("upperarm", config.complex_split.bones)
        self.assertNotIn("J_Sec_Hair", config.bone_filter.merge_to_parent_name_contains)
        self.assertNotIn("J_Adj_", config.bone_filter.merge_to_parent_name_contains)


def write_two_bone_fixture(path: Path) -> None:
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
    payload = base64.b64encode(joints + weights).decode("ascii")
    model = {
        "asset": {"version": "2.0", "generator": "gltf2bb partition test"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "child_joint"},
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
            {"componentType": 5126, "count": 6, "type": "VEC3"},
            {"bufferView": 0, "componentType": 5121, "count": 6, "type": "VEC4"},
            {"bufferView": 1, "componentType": 5126, "count": 6, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(joints), "byteLength": len(weights)},
        ],
        "buffers": [
            {
                "byteLength": len(joints) + len(weights),
                "uri": f"data:application/octet-stream;base64,{payload}",
            }
        ],
    }
    path.write_text(json.dumps(model), encoding="utf-8")


def write_helper_merge_fixture(path: Path) -> None:
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
        "asset": {"version": "2.0", "generator": "gltf2bb helper merge test"},
        "scene": 0,
        "scenes": [{"nodes": [3]}],
        "nodes": [
            {"name": "root_joint", "children": [1]},
            {"name": "twist_helper_joint", "children": [2]},
            {"name": "child_joint"},
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


if __name__ == "__main__":
    unittest.main()
