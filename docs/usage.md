# gltf2bb Usage Guide

`gltf2bb` converts glTF, GLB, or VRM models with skeletons and skin weights into Blockbench-friendly `.bbmodel` project files. The project is still an MVP: it prioritizes reading input models, assigning faces to bones, generating editable cube hierarchies, and preserving usable pivots. Output is cube-only: the writer emits Blockbench free-model cube elements, groups, and outliner data. It does not emit arbitrary `.bbmodel` mesh elements, UV maps, texture assets, materials, or animation data.

## Requirements

- Run project commands with `uv`.
- Python `>=3.14` is required by the current project metadata.
- There are currently no external Python dependencies.
- Supported input formats are `.gltf`, `.glb`, and `.vrm`.
- Config files must be JSON. YAML is not supported.

## Quick Start

Show CLI help:

```bash
uv run gltf2bb --help
```

Inspect model statistics:

```bash
uv run gltf2bb inspect path/to/model.glb
```

Write a bone partition report:

```bash
uv run gltf2bb partition path/to/model.glb --preset mmd_humanoid --report out/partition-report.json
```

Convert to a Blockbench project:

```bash
uv run gltf2bb convert path/to/model.glb -o out/model.bbmodel --preset mmd_humanoid --report out/convert-report.json
```

If `-o` / `--output` is omitted, `convert` writes to `out/<input-stem>.bbmodel`.

## Recommended Workflow

1. Run `inspect` first to confirm the model has `skins`, `JOINTS_0`, and `WEIGHTS_0`.
2. Run `partition` to verify that faces can be assigned to dominant bones.
3. Start with `convert --mode cuboid` for the most conservative one-bbox-cube-per-bone output.
4. For character models, try `--preset mmd_humanoid` or `--preset humanoid`.
5. For hair, skirts, coats, accessories, or other complex parts, try `--mode hybrid` or `--complex-split`.
6. Open the `.bbmodel` in Blockbench and manually refine proportions, cube shapes, joints, and any animations you choose to create there.

## Commands

### inspect

Prints model, skeleton, and skin weight statistics. Use it to check whether the input model is suitable for conversion.

```bash
uv run gltf2bb inspect path/to/model.glb
```

Useful fields to check:

- `Skins` should be greater than 0.
- `Skin joints` and `Unique joint nodes` should look reasonable for the model.
- `Vertices with JOINTS_0` should be close to the total vertex count.
- `Weighted vertices` should be close to the total vertex count.
- `Warnings` may reveal missing accessors, missing skins, or invalid joint references.

### partition

Assigns each face to the resolved bone with the highest accumulated skin weight, then writes a partition report.

```bash
uv run gltf2bb partition path/to/model.glb --report out/report.json
```

Common options:

- `--preset default|humanoid|mmd_humanoid`: Select a built-in bone filtering preset.
- `--config config.json`: Override the selected preset with a JSON config file.
- `--report out/report.json`: Write a JSON report. If omitted, the report JSON is printed to stdout.

Important report fields:

- `totals.assigned_faces` should be close to `totals.faces`.
- `totals.fallback_faces` should not be unexpectedly high.
- `totals.unassigned_faces` should usually be 0 or close to 0.
- `bone_resolution.merged_to_parent` shows helper bones that were merged into kept parents.
- `bones[].faces` shows which kept bones own geometry.

### convert

Generates a Blockbench `.bbmodel` file. Current output consists of cube elements, groups, and outliner hierarchy in Blockbench free-model format. Arbitrary mesh element output, UV export, texture export, material export, and animation export are out of scope for the current implementation and are not emitted.

```bash
uv run gltf2bb convert path/to/model.glb -o out/model.bbmodel
```

Common options:

- `-o, --output out/model.bbmodel`: Output path. Defaults to `out/<input-stem>.bbmodel`.
- `--mode cuboid|hybrid`: Conversion mode. Defaults to `cuboid`.
- `--target-height 32`: Scale the assigned mesh bbox to this height in Blockbench units. Defaults to `32`.
- `--preset default|humanoid|mmd_humanoid`: Select a built-in bone filtering preset.
- `--config config.json`: Override the selected preset with a JSON config file.
- `--report out/convert-report.json`: Write a conversion report.
- `--complex-split head`: Split a matching complex bone into multiple editable cubes. This option can be repeated.

Examples:

```bash
uv run gltf2bb convert character.glb --mode cuboid --preset mmd_humanoid --report out/convert-report.json
```

```bash
uv run gltf2bb convert character.glb --mode hybrid --preset mmd_humanoid --report out/hybrid-report.json
```

```bash
uv run gltf2bb convert character.vrm --preset mmd_humanoid --complex-split head --report out/head-report.json
```

## Conversion Modes

### cuboid

`cuboid` is the default mode. It creates one bbox cube for each resolved bone that owns faces, and it creates Blockbench groups from the skeleton hierarchy.

Use it for:

- Validating the full conversion pipeline.
- Coarse Minecraft-style or Blockbench-style models.
- Debugging bone assignment, pivots, scaling, and hierarchy output.

Limitations:

- A single complex bone can still become one large bbox cube.
- Hair, skirts, accessories, and decorations may be very coarse.

### hybrid

`hybrid` does not emit mesh elements. It remains cube-only and automatically enables special cube splitting for complex bones such as `head`, `hair`, `skirt`, `coat`, and `accessory`. Other bones keep the normal bbox cube behavior.

```bash
uv run gltf2bb convert character.glb --mode hybrid --preset mmd_humanoid --report out/hybrid-report.json
```

The report records this behavior in fields such as:

- `hybrid.mesh_strategy = "special_cubes"`
- `complex_split.subparts[].method`
- `totals.complex_split_bones`
- `totals.hybrid_special_cube_bones`

## Presets

Available built-in presets:

- `default`: No extra name-based bone filtering rules.
- `humanoid`: Filters common humanoid helper bones such as `IK`, `physics`, `control`, and `ctrl`; merges bones containing `twist`, `helper`, `secondary`, or `spring` into the nearest kept parent.
- `mmd_humanoid`: Extends `humanoid` with common MMD helper, physics, auxiliary, and roll bone name patterns.

Examples:

```bash
uv run gltf2bb partition character.glb --preset mmd_humanoid --report out/partition-report.json
uv run gltf2bb convert character.glb --preset mmd_humanoid --report out/convert-report.json
```

## JSON Config

Config files override the selected preset. Only JSON is supported.

### Disable Bone Filtering

```json
{
  "bone_filter": {
    "enabled": false
  }
}
```

Use it with:

```bash
uv run gltf2bb convert character.glb --config config.json --report out/report.json
```

### Custom Bone Merge Rules

```json
{
  "bone_merge": {
    "merge_to_parent_name_contains": ["twist", "helper", "spring", "aux", "roll"],
    "merge_to_parent_name_regex": [".*_end$"],
    "case_sensitive": false,
    "report_merged_bones": true
  }
}
```

### Complex Bone Splitting

```json
{
  "complex_split": {
    "enabled": true,
    "bones": ["head", "hair", "skirt"],
    "connected_components": {
      "enabled": true,
      "min_faces": 8,
      "merge_tiny_components_to_nearest": true,
      "delete_tiny_components": false
    }
  }
}
```

Notes:

- `bones` matches complex bones by name or by built-in aliases.
- `head` uses material names, connected components, and spatial rules to infer parts such as `head_core`, `hair_*`, `eye_*`, `brow_*`, `mouth`, `nose`, and `ear_*`.
- Non-head complex bones use connected components to create multiple editable cubes when possible.
- VRM 1.0 heads can be detected through `extensions.VRMC_vrm.humanoid.humanBones.head.node` even if the node name does not contain `head`.

### Small Part Cleanup

```json
{
  "cleanup": {
    "delete_small_parts": true,
    "merge_small_parts_to_parent": false,
    "min_faces": 8,
    "min_bbox_volume": 0.01
  }
}
```

Notes:

- `delete_small_parts` removes parts below the configured thresholds.
- `merge_small_parts_to_parent` merges matching parts into the nearest kept parent bone instead of deleting them.
- Cleanup results are recorded under `cleanup`, `totals.deleted_small_parts`, and `totals.merged_small_parts` in the convert report.

### Experimental Oriented Cubes

```json
{
  "oriented_cubes": {
    "enabled": true,
    "bones": ["head", "Thumb"],
    "scope": "matching_bones"
  }
}
```

Supported `scope` values:

- `complex_split_parts`: Only affects cubes generated by complex split.
- `bone_cubes`: Affects normal bone cubes.
- `matching_bones`: Affects cubes whose owner bone name matches the configured names.

This feature is experimental and disabled by default. Results are recorded under `oriented_cubes`, `orientation_decisions`, and `totals.oriented_cubes` in the convert report. Each `oriented_cubes[]` entry includes the cube name, owner bone, rotation, source, reason, original and oriented bbox volume when available, and `cube_only_compatible`. `orientation_decisions[]` records accepted and rejected auto-orientation candidates so rejected rotations can be audited without changing the cube-only output.

### Face Feature Protection

`face_feature_protection` is enabled by default. It adjusts `head_core` and `hair_front` cube bounds around explicit eye, brow, mouth, nose, and similar face-feature parts when those parts are present.

```json
{
  "face_feature_protection": {
    "enabled": true,
    "min_faces": 32,
    "margin_ratio": 0.002,
    "outlier_gap_ratio": 0.05,
    "protect_hair_front": true,
    "protect_head_core_front": true
  }
}
```

Report actions are written to `face_feature_protection.actions`. Each action records `adjusted_part_name`, `cube_name`, before and after bbox values, `protected_feature_names`, `axis`, `target_value`, `margin`, `front_sign`, and `overlap_axes`.

## Output Files

Current `.bbmodel` output includes:

- Blockbench free model metadata.
- Cube elements.
- Groups corresponding to kept skeleton bones.
- Group and cube outliner hierarchy.
- Empty `textures` array.
- Empty `animations` array.

The empty arrays keep the project shape familiar to Blockbench, but `gltf2bb` does not export source textures, UV maps, materials, arbitrary mesh elements, or animation clips.

Coordinate handling:

- The assigned mesh bbox is scaled to `--target-height`.
- The model is centered on XZ.
- The model is grounded on Y.
- Group origins come from bone node world translations.
- Mesh node transforms are applied to `POSITION` before bboxes are computed.

## Report Reference

### Partition Report

Key fields:

- `totals.faces`: Number of triangle faces processed.
- `totals.assigned_faces`: Number of faces assigned to an owner bone.
- `totals.fallback_faces`: Number of faces assigned through fallback because weights were missing or unusable.
- `totals.unassigned_faces`: Number of faces that could not be assigned.
- `bone_resolution`: Bone filtering, merge, and ignore decisions.
- `bones`: Face and vertex counts for each kept bone.
- `warnings`: Input and processing diagnostics.

### Convert Report

Key fields:

- `output`: Generated `.bbmodel` path.
- `scale`: Scale applied to the output model.
- `totals.cubes`: Number of generated cubes.
- `totals.empty_bones`: Number of kept bones with no geometry.
- `complex_split`: Details for complex bone splitting. Its budget fields include `original_cube_dimensions`, `split_method`, `requested_subpart_count`, `budget_limit`, `budget_status`, and `budget_reason`.
- `cleanup`: Small part deletion, merge, and keep records.
- `oriented_cubes`: Cubes that received rotation, including `reason`, `original_bbox_volume`, `oriented_bbox_volume`, and `cube_only_compatible`.
- `orientation_decisions`: Accepted and rejected auto-orientation candidates.
- `face_feature_protection.actions`: Face-feature protection adjustments with adjusted part name, protected feature names, axis, target value, margin, front sign, and overlap axes.
- `quality`: Cube-only output diagnostics.
- `warnings`: Conversion diagnostics.

Important `quality` fields:

- `quality.cube_only`: Confirms this report came from a cube-only `.bbmodel` export. It includes cube count and zero mesh/vertex element counts.
- `quality.cube_count_by_owner_bone`: Cube counts grouped by resolved owner bone.
- `quality.largest_cubes`: Largest cube candidates by volume for quick inspection.
- `quality.oversized_cubes`: Largest-by-volume cubes flagged as oversized candidates.
- `quality.unrotated_elongated_cubes`: Long cubes that did not receive rotation, with a reason.
- `quality.tiny_fragment_cubes`: Small low-face cube candidates that may be cleanup noise.
- `quality.skipped_unskinned_meshes_summary`: Count, node indices, mesh indices, and reasons for skipped unskinned meshes.
- `quality.split_diagnostics`: Per-complex-bone split methods, output cube count, tiny component handling, and budget fields.
- `quality.cube_budget_warnings`: Model-level, owner-bone-level, and split budget warnings such as `cube_count_exceeds_budget`, `owner_cube_count_exceeds_budget`, `auto_spatial_split_capped_to_budget`, and `regular_detail_split_exceeds_owner_budget`.

## Input Model Recommendations

Ideal input:

- One or more skinned meshes.
- A clear skeleton.
- Useful bone names.
- Valid `JOINTS_0` and `WEIGHTS_0` data.
- Rest pose, A-pose, or T-pose geometry.
- No complex constraints, or constraints already baked into normal glTF skinning.

Poor input:

- Static mesh only, with no skin.
- Skeleton or weights lost during export.
- Many vertices with zero weights.
- A model split into many unnamed meshes.
- Multiple overlapping armatures with confusing hierarchy.

## Troubleshooting

### `inspect` Shows `Skins: 0`

The input does not contain glTF skin data. This tool cannot reliably split such a model by bone. Export the source model again with skeletons and skin weights preserved.

### `fallback_faces` Is High

This usually means `JOINTS_0` or `WEIGHTS_0` is missing or unreadable. Check `Warnings` for accessor, buffer, and `skin.joints` problems.

### Generated Cubes Are Too Large

A single bone may own complex geometry such as hair, a skirt, a cape, or accessories. Try hybrid mode:

```bash
uv run gltf2bb convert character.glb --mode hybrid --preset mmd_humanoid --report out/hybrid-report.json
```

Or explicitly split complex bones:

```bash
uv run gltf2bb convert character.glb --complex-split head --complex-split hair --report out/report.json
```

### Output Scale Is Wrong

Adjust `--target-height`:

```bash
uv run gltf2bb convert character.glb --target-height 24 -o out/model.bbmodel
```

### Config File Fails To Load

Only JSON config files are supported. Do not use YAML or planned-but-unimplemented config fields.

## Current Limitations

- PMX and FBX are not read directly.
- `.vrm` files are read as GLB, with limited VRM 1.0 humanoid detection.
- Textures, UVs, and animations are not exported.
- `.bbmodel` mesh elements are not exported.
- Artistic proportions are not automatically fixed.
- PMX-specific IK, rigid body, physics, and morph semantics are not restored.
- `hybrid` currently means special cube splitting, not true mesh/cuboid mixed export.

## Development And Verification

Run the test suite:

```bash
uv run python -m unittest discover -s tests
```

Run the exp-model quality benchmark helper when the ignored `exp/` models are available:

```bash
uv run python tests/quality_benchmark.py --evidence .omo/evidence/quality-benchmark-summary.json
```

The helper converts the selected exp models in `hybrid` mode and writes raw `.bbmodel` plus raw report JSON files to `/tmp/gltf2bb-quality/`. Keep those raw files out of the repo. The evidence file is a compact summary with command status, cube count, report quality keys, totals keys, and budget warnings for each model.

Run a minimal fixture through the full pipeline:

```bash
uv run gltf2bb inspect tests/fixtures/minimal_skinned_triangle.gltf
uv run gltf2bb partition tests/fixtures/minimal_skinned_triangle.gltf --report out/report.json
uv run gltf2bb convert tests/fixtures/minimal_skinned_triangle.gltf -o out/model.bbmodel --mode cuboid --report out/convert-report.json
```
