# PMX Models

`gltf2bb` does not read `*.pmx` files directly. Convert PMX/MMD models to glTF or GLB first, then run `inspect`, `partition`, and `convert` on the exported file.

PMX-specific data such as IK, rigid bodies, physics joints, and some morph semantics will not be restored by this project. The goal is to preserve enough skeleton, skin weights, mesh geometry, and material names to generate a Blockbench-friendly `.bbmodel`.

## Requirements

- [Blender 4.2 LTS or above](https://blender.org)
- [MMD Tools](https://extensions.blender.org/add-ons/mmd-tools/)
- A `*.pmx` model with an MMD armature, skin weights, and useful bone/material names

## Export From Blender

1. Open Blender.
2. Delete the default Cube, Camera, and Light.
3. Import the PMX model with MMD Tools. Dragging the `*.pmx` file into the Blender window is usually enough; keep the default import operator settings unless the model needs special handling.
4. Confirm the model has an armature and skinned mesh. If the armature or weights are missing here, `gltf2bb` cannot reliably split the model by bone later.
5. Open `File` -> `Export` -> `glTF 2.0 (.glb/.gltf)`.
6. Prefer `.glb` for a single portable file. `.gltf` is also supported when its external `.bin` and texture files stay next to it.
7. Keep skeleton, skinning, and mesh data enabled in Blender's glTF export options.
8. Export the file.

## Verify The Export

Run `inspect` first:

```bash
uv run gltf2bb inspect path/to/model.glb
```

Check that the output shows:

- `Skins` greater than `0`.
- A reasonable number of skin joints.
- Most vertices have `JOINTS_0` and `WEIGHTS_0`.

Then generate a partition report:

```bash
uv run gltf2bb partition path/to/model.glb --preset mmd_humanoid --report out/partition-report.json
```

The report should show most faces assigned to bones and a low number of fallback or unassigned faces.

## Convert To Blockbench

Start with conservative cuboid output:

```bash
uv run gltf2bb convert path/to/model.glb -o out/model.bbmodel --mode cuboid --preset mmd_humanoid --report out/convert-report.json
```

For PMX character models with hair, skirts, coats, accessories, or large single-bone parts, try hybrid mode:

```bash
uv run gltf2bb convert path/to/model.glb -o out/model.bbmodel --mode hybrid --preset mmd_humanoid --report out/hybrid-report.json
```

If the head becomes one large cube, explicitly enable head splitting:

```bash
uv run gltf2bb convert path/to/model.glb -o out/model.bbmodel --preset mmd_humanoid --complex-split head --report out/head-report.json
```

## Troubleshooting

- `Skins: 0`: The Blender export did not include glTF skin data. Re-export with armature and skinning preserved.
- Many missing or zero weights: Check the PMX import result in Blender before exporting. The mesh may not be bound to the armature correctly.
- Huge cubes around hair, skirts, or accessories: Use `--mode hybrid` or repeat `--complex-split` for matching bones such as `head`, `hair`, `skirt`, or `accessory`.
- Bone hierarchy looks noisy: Use `--preset mmd_humanoid` so common MMD helper, IK, physics, and roll bones are filtered or merged into parent bones.
- `.gltf` cannot be loaded: Make sure the `.gltf` file and its referenced `.bin` buffer file are kept together. Export `.glb` if in doubt.
