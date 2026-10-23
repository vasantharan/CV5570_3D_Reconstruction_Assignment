from pathlib import Path
import csv
import math
import sys
import numpy as np
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

COLMAP_MODEL_DIR = Path(r"D:\CV5570\CV5570_3D_Reconstruction_Assignment\Reconstruction Workspace\sparse\0")
OUTPUT_DIR = Path(r"D:\CV5570\CV5570_3D_Reconstruction_Assignment\Reconstruction Workspace\output")

def parse_cameras_txt(path):
    cameras = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            camera_id = int(p[0])
            model = p[1]
            width, height = int(p[2]), int(p[3])
            params = np.array([float(v) for v in p[4:]], dtype=float)
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def make_intrinsic_matrix(model, p):
    if model in (
        "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
        "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"
    ):
        f, cx, cy = p[:3]
        fx = fy = f
    elif model in (
        "PINHOLE", "OPENCV", "FULL_OPENCV", "FOV",
        "THIN_PRISM_FISHEYE", "OPENCV_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE"
    ):
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(
            f"Unsupported camera model {model!r}. "
            "This script will not guess its intrinsic parameter layout."
        )
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def quaternion_to_rotation(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=float)


def parse_images_txt(path):
    images = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:
            raise ValueError(f"Could not parse images.txt metadata line: {line}")
        image_id = int(p[0])
        q = np.array([float(v) for v in p[1:5]], dtype=float)
        t = np.array([float(v) for v in p[5:8]], dtype=float)
        camera_id = int(p[8])
        name = " ".join(p[9:])
        R = quaternion_to_rotation(q)
        Rt = np.column_stack((R, t))
        C = -R.T @ t
        images.append({
            "image_id": image_id,
            "name": name,
            "camera_id": camera_id,
            "qvec": q,
            "tvec": t,
            "R": R,
            "Rt": Rt,
            "C": C,
        })
        if i < len(lines):
            i += 1
    return images

def fmt_matrix(a, precision=8):
    return np.array2string(a, precision=precision, suppress_small=False)

def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def add_matrix_table(doc, matrix, fmt="{:.8f}"):
    table = doc.add_table(rows=matrix.shape[0], cols=matrix.shape[1])
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            cell = table.cell(r, c)
            cell.text = fmt.format(float(matrix[r, c]))
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    return table

def main():
    cameras_file = COLMAP_MODEL_DIR / "cameras.txt"
    images_file = COLMAP_MODEL_DIR / "images.txt"
    if not cameras_file.is_file() or not images_file.is_file():
        print("ERROR: Could not find both cameras.txt and images.txt.")
        print(f"Checked model folder: {COLMAP_MODEL_DIR}")
        print("If your model contains .bin files, export the sparse model to TXT in COLMAP.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cameras = parse_cameras_txt(cameras_file)
    images = parse_images_txt(images_file)

    if not cameras:
        sys.exit("No camera calibration records were found in cameras.txt.")
    if not images:
        sys.exit("No registered image poses were found in images.txt.")

    txt_path = OUTPUT_DIR / "camera_calibration_report.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("CV5570 — 3D RECONSTRUCTION: CAMERA CALIBRATION AND CAMERA POSITIONS\n")
        f.write("=" * 72 + "\n\n")
        f.write("Source: COLMAP cameras.txt and images.txt. Values are read from the model.\n")
        f.write("COLMAP pose convention: X_camera = R X_world + t.\n")
        f.write("Camera centre in world coordinates: C = -R^T t.\n\n")
        f.write("1. INTRINSIC CAMERA MATRICES K\n\n")
        for cid, cam in sorted(cameras.items()):
            K = make_intrinsic_matrix(cam["model"], cam["params"])
            f.write(f"Camera ID: {cid}\nModel: {cam['model']}\n")
            f.write(f"Image resolution: {cam['width']} x {cam['height']} pixels\n")
            f.write(f"COLMAP parameters: {cam['params'].tolist()}\n")
            f.write("K =\n" + fmt_matrix(K) + "\n")
            distortion = cam["params"][3:] if cam["model"] in (
                "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "FOV",
                "THIN_PRISM_FISHEYE", "OPENCV_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE",
                "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"
            ) else []
            f.write(f"Additional distortion parameters (if any): {list(distortion)}\n\n")
        f.write("\n2. EXTRINSIC CAMERA MATRICES [R|t] AND CAMERA POSITIONS\n\n")
        for im in images:
            f.write(f"Image ID: {im['image_id']}\nImage name: {im['name']}\n")
            f.write(f"Camera ID: {im['camera_id']}\n")
            f.write("[R|t] =\n" + fmt_matrix(im["Rt"]) + "\n")
            f.write("Camera centre C = -R^T t (COLMAP world coordinates) =\n")
            f.write(fmt_matrix(im["C"].reshape(1, 3)) + "\n\n")

    intrinsic_rows = []
    for cid, cam in sorted(cameras.items()):
        K = make_intrinsic_matrix(cam["model"], cam["params"])
        intrinsic_rows.append([
            cid, cam["model"], cam["width"], cam["height"],
            *cam["params"].tolist(),
            K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        ])
    max_params = max(len(c["params"]) for c in cameras.values())
    intrinsic_header = ["camera_id", "model", "width_px", "height_px"] + [
        f"colmap_param_{i+1}" for i in range(max_params)
    ] + ["fx_px", "fy_px", "cx_px", "cy_px"]
    
    intrinsic_rows_padded = []
    for row in intrinsic_rows:
        fixed = row[:4]
        param_count = len(cameras[row[0]]["params"])
        params = row[4:4 + param_count]
        params += [""] * (max_params - param_count)
        K_tail = row[4 + param_count:]
        intrinsic_rows_padded.append(fixed + params + K_tail)
    write_csv(OUTPUT_DIR / "camera_intrinsics.csv", intrinsic_header, intrinsic_rows_padded)

    position_rows = []
    extrinsic_rows = []
    for im in images:
        position_rows.append([
            im["image_id"], im["name"], im["camera_id"],
            *im["C"].tolist()
        ])
        flat = im["Rt"].reshape(-1).tolist()
        extrinsic_rows.append([im["image_id"], im["name"], im["camera_id"], *flat])
    write_csv(
        OUTPUT_DIR / "camera_positions.csv",
        ["image_id", "image_name", "camera_id", "camera_x_world", "camera_y_world", "camera_z_world"],
        position_rows
    )
    rt_names = [f"r{r+1}{c+1}" for r in range(3) for c in range(3)] + ["tx", "ty", "tz"]
    write_csv(
        OUTPUT_DIR / "camera_extrinsics.csv",
        ["image_id", "image_name", "camera_id"] + rt_names,
        extrinsic_rows
    )

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.7)
    sec.right_margin = Inches(0.7)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(9)

    title = doc.add_heading("3D Reconstruction: Camera Calibration and Camera Positions", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("CV5570 — Computer Vision for Construction Engineering")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Data source: the registered COLMAP sparse model. Intrinsic and extrinsic values below "
        "are calculated from the supplied cameras.txt and images.txt files; they are not assumed."
    )
    doc.add_heading("1. Intrinsic Camera Matrix [K]", level=1)
    doc.add_paragraph(
        "The intrinsic matrix describes focal lengths in pixels and the principal point. "
        "For each COLMAP camera, K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. "
        "Lens distortion parameters, when present, are listed separately."
    )
    for cid, cam in sorted(cameras.items()):
        K = make_intrinsic_matrix(cam["model"], cam["params"])
        doc.add_heading(f"Camera ID {cid}", level=2)
        doc.add_paragraph(
            f"Model: {cam['model']} | Image resolution: {cam['width']} × {cam['height']} pixels"
        )
        doc.add_paragraph(f"Raw COLMAP parameters: {cam['params'].tolist()}")
        add_matrix_table(doc, K)
        if len(cam["params"]) > (3 if cam["model"] in (
            "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
            "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"
        ) else 4):
            doc.add_paragraph(
                f"Additional distortion parameters: {cam['params'][3:].tolist()}"
            )

    doc.add_heading("2. Extrinsic Camera Matrices [R|t] and Camera Positions", level=1)
    doc.add_paragraph(
        "COLMAP uses X_camera = R X_world + t. The camera centre in the COLMAP world "
        "coordinate system is C = −Rᵀt. Translation t is not itself the camera centre."
    )
    for idx, im in enumerate(images):
        doc.add_heading(f"Image {im['image_id']}: {im['name']}", level=2)
        doc.add_paragraph(f"Camera ID: {im['camera_id']}")
        add_matrix_table(doc, im["Rt"])
        doc.add_paragraph(
            "Camera centre C = −Rᵀt (COLMAP world coordinates): "
            + np.array2string(im["C"], precision=8)
        )
        if idx != len(images) - 1:
            doc.add_paragraph("")

    doc.add_heading("3. Output notes", level=1)
    doc.add_paragraph(
        f"Number of camera calibration records: {len(cameras)}. "
        f"Number of registered image poses: {len(images)}."
    )
    doc.add_paragraph(
        "COLMAP reconstruction coordinates are generally in an arbitrary scale unless the model "
        "has been metrically scaled. Camera centres therefore use the reconstruction's world units."
    )
    docx_path = OUTPUT_DIR / "camera_calibration_report.docx"
    doc.save(docx_path)

    print("Camera calibration export completed.")
    print(f"Cameras read: {len(cameras)}")
    print(f"Registered image poses read: {len(images)}")
    print(f"Word report: {docx_path}")
    print(f"Text report: {txt_path}")
    print(f"Intrinsics CSV: {OUTPUT_DIR / 'camera_intrinsics.csv'}")
    print(f"Camera positions CSV: {OUTPUT_DIR / 'camera_positions.csv'}")
    print(f"Extrinsics CSV: {OUTPUT_DIR / 'camera_extrinsics.csv'}")


if __name__ == "__main__":
    main()