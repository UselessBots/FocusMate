import numpy as np
import yourdfpy          # pip install yourdfpy trimesh[easy]
import rhino3dm          # pip install rhino3dm


def urdf_cfg_to_3dm(
    urdf_path: str,
    configuration,
    out_3dm_path: str,
    *,
    use_collision_geometry: bool = False,
    mesh_dir: str | None = None,
) -> None:
    """
    Export the full robot shape at a given configuration as a Rhino .3dm file.

    Args:
        urdf_path: Path to URDF file.
        configuration:
            - dict: {joint_name: value}, or
            - list/tuple/np.ndarray: values in order of URDF.actuated_joint_names
              (yourdfpy handles both).
        out_3dm_path: Output .3dm file path.
        use_collision_geometry: If True, use <collision> meshes instead of <visual>.
        mesh_dir: Optional root directory for mesh files (package://, relative paths, etc.).
    """
    load_kwargs = {}
    if mesh_dir is not None:
        load_kwargs["mesh_dir"] = mesh_dir

    # Load URDF and build scene graphs so we can do FK and get trimesh scenes. :contentReference[oaicite:0]{index=0}
    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        build_collision_scene_graph=use_collision_geometry,
        load_meshes=not use_collision_geometry or True,   # always load visual meshes when needed
        load_collision_meshes=use_collision_geometry,
        **load_kwargs,
    )

    # Apply joint configuration (does FK and updates internal scene transforms). :contentReference[oaicite:1]{index=1}
    urdf.update_cfg(configuration)

    # Pick which geometry to export. Both are trimesh.Scene objects. :contentReference[oaicite:2]{index=2}
    scene = urdf.collision_scene if use_collision_geometry else urdf.scene
    if scene is None or len(scene.geometry) == 0:
        raise RuntimeError(
            "URDF scene has no geometry. Check load_meshes/load_collision_meshes and mesh_dir."
        )

    # Bake all link meshes into a single Trimesh, with transforms applied. :contentReference[oaicite:3]{index=3}
    mesh = scene.to_mesh()

    # ---- Convert Trimesh -> rhino3dm.Mesh ----
    rh_mesh = rhino3dm.Mesh()

    # Vertices
    for vx, vy, vz in mesh.vertices:
        rh_mesh.Vertices.Add(float(vx), float(vy), float(vz))

    # Faces (Trimesh is usually triangles, but we guard for quads)
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2:
        raise RuntimeError(f"Unexpected faces array shape {faces.shape}")
    if faces.shape[1] == 3:
        for i0, i1, i2 in faces:
            rh_mesh.Faces.AddFace(int(i0), int(i1), int(i2))
    elif faces.shape[1] == 4:
        for i0, i1, i2, i3 in faces:
            rh_mesh.Faces.AddFace(int(i0), int(i1), int(i2), int(i3))
    else:
        raise RuntimeError(
            "Only triangle/quad faces are supported; got faces with "
            f"{faces.shape[1]} vertices."
        )

    # Optional, if you care about shading normals
    # rh_mesh.Normals.ComputeNormals()

    rh_mesh.Compact()  # clean up internal arrays :contentReference[oaicite:4]{index=4}

    # ---- Write .3dm ----
    model = rhino3dm.File3dm()
    # Add as a single mesh object; you could also add per-link meshes if you want. :contentReference[oaicite:5]{index=5}
    model.Objects.AddMesh(rh_mesh)
    # version: 0 = auto, 7/8 = specific Rhino version
    model.Write(out_3dm_path, 8)


import os
from typing import Mapping, Sequence, Union

import numpy as np
import yourdfpy          # pip install yourdfpy trimesh[easy]
import trimesh


def _normalize_cfg(
    urdf: yourdfpy.URDF,
    configuration: Union[Mapping[str, float], Sequence[float], np.ndarray],
) -> np.ndarray:
    """
    Normalize configuration into a 1D array in urdf.actuated_joint_names order.

    - If configuration is a dict: {joint_name: value}. Missing joints -> 0.0.
    - If configuration is a sequence: assumed in actuated_joint_names order.
    """
    actuated_names = urdf.actuated_joint_names

    if isinstance(configuration, Mapping):
        q = []
        for name in actuated_names:
            val = configuration.get(name, 0.0)
            q.append(float(val))
        return np.asarray(q, dtype=float)

    # sequence / array
    cfg_arr = np.asarray(configuration, dtype=float).reshape(-1)
    if cfg_arr.shape[0] != len(actuated_names):
        raise ValueError(
            f"Expected {len(actuated_names)} joint values, got {cfg_arr.shape[0]}"
        )
    return cfg_arr


def urdf_cfg_to_obj(
    urdf_path: str,
    configuration: Union[Mapping[str, float], Sequence[float], np.ndarray],
    out_obj_path: str,
    *,
    use_collision_geometry: bool = False,
    mesh_dir: str | None = None,
    merge_meshes: bool = True,
) -> None:
    """
    Export the robot shape at a given configuration as a single OBJ mesh.

    Args:
        urdf_path: Path to URDF file.
        configuration:
            - dict: {joint_name: value}, or
            - sequence/np.ndarray: values in urdf.actuated_joint_names order.
        out_obj_path: Output .obj path.
        use_collision_geometry: If True, use <collision> meshes instead of <visual>.
        mesh_dir: Optional mesh root directory (for package:// or relative paths).
        merge_meshes:
            If True, bake scene to a single Trimesh and export one OBJ.
            If False, let trimesh.Scene export (multiple objects/groups).
    """
    load_kwargs = {}
    if mesh_dir is not None:
        load_kwargs["mesh_dir"] = mesh_dir

    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        build_collision_scene_graph=use_collision_geometry,
        load_meshes=not use_collision_geometry or True,
        load_collision_meshes=use_collision_geometry,
        **load_kwargs,
    )

    cfg_vec = _normalize_cfg(urdf, configuration)
    urdf.update_cfg(cfg_vec)

    scene = urdf.collision_scene if use_collision_geometry else urdf.scene
    if scene is None or len(scene.geometry) == 0:
        raise RuntimeError(
            "URDF scene has no geometry; check meshes, mesh_dir, and load_* flags."
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_obj_path)), exist_ok=True)

    if merge_meshes:
        # Single baked mesh
        mesh: trimesh.Trimesh = scene.to_mesh()
        mesh.export(out_obj_path)
    else:
        # Multi-object OBJ; groups are preserved as separate objects
        scene.export(out_obj_path, file_type="obj")

    print(f"Saved robot OBJ model to: {out_obj_path}")

if __name__ == "__main__":
    # Example usage
    urdf_path = "follower.urdf"
    joint_positions = {
        "joint_1": 0.0,
        "joint_2": -0.4,
        "joint_3": 0.4,
        "joint_4": -0.9,
        "joint_5": 0.0,
        "joint_6": 0.0,
    }
    output_3dm_path = "output/robot_model.3dm"

    urdf_cfg_to_3dm(urdf_path, joint_positions, "robot.3dm")
    urdf_cfg_to_obj(urdf_path, joint_positions, "robot.obj")