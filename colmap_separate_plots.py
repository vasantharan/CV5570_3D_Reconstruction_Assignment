
from pathlib import Path
import open3d as o3d
import numpy as np

PROJECT_DIR = Path(
    r"D:\CV5570\CV5570_3D_Reconstruction_Assignment\Reconstruction Workspace"
)
OUTPUT_DIR = PROJECT_DIR / "output"

FILES = [
    (
        PROJECT_DIR / "dense" / "0" / "fused.ply",
        "original_point_cloud_colmap_style.png",
    ),
    (
        OUTPUT_DIR / "point_cloud_scaled_rotated_45deg_x.ply",
        "scaled_rotated_point_cloud_colmap_style.png",
    ),
]

BACKGROUND = np.array([0.025, 0.19, 0.29])
WIDTH, HEIGHT = 1600, 1000


def save_colmap_style_plot(ply_path, output_path):
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    cloud = o3d.io.read_point_cloud(str(ply_path))

    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")
    vis = o3d.visualization.Visualizer()

    if not vis.create_window(
        window_name=ply_path.name,
        width=WIDTH,
        height=HEIGHT,
        visible=True,
    ):
        raise RuntimeError("Could not create Open3D visualization window.")

    vis.add_geometry(cloud)

    bounds = cloud.get_axis_aligned_bounding_box()
    size = max(float(np.max(bounds.get_extent())), 1e-6)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size * 0.12,
        origin=bounds.get_center(),
    )
    vis.add_geometry(axes)

    render = vis.get_render_option()
    render.background_color = BACKGROUND
    render.point_size = 2.0
    render.show_coordinate_frame = True

    vis.reset_view_point(True)
    view = vis.get_view_control()

    view.set_lookat(bounds.get_center())
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_zoom(0.75)

    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(output_path), do_render=True)
    vis.destroy_window()

    print(f"Saved: {output_path}")


def main():
    for ply_path, filename in FILES:
        save_colmap_style_plot(
            ply_path,
            OUTPUT_DIR / filename,
        )

    print("\nBoth COLMAP-style screenshots are saved.")


if __name__ == "__main__":
    main()