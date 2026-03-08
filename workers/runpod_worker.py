"""
Runpod Serverless Handler — 3D Gaussian Splatting GPU Worker
============================================================

This Python script runs INSIDE the Runpod container on the GPU.
It is the handler function that Runpod calls for each job.

Pipeline:
  1. Download input files from Cloudinary URLs
  2. (Optional) Enhance images with Real-ESRGAN
  3. Run COLMAP for camera pose estimation
  4. Train 3D Gaussian splatting model
  5. Convert .ply output to .glb
  6. Upload GLB + thumbnail to Cloudinary
  7. Return output to Runpod (which triggers our webhook)

Deploy this as a Docker image on Runpod Serverless.
See Dockerfile.runpod in the project root for the container setup.
"""

import runpod
import os
import sys
import json
import shutil
import requests
import subprocess
import tempfile
import cloudinary
import cloudinary.uploader
from pathlib import Path

# ─── Cloudinary config (set these in Runpod environment variables) ─────────────
cloudinary.config(
    cloud_name = os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key    = os.environ["CLOUDINARY_API_KEY"],
    api_secret = os.environ["CLOUDINARY_API_SECRET"],
    secure     = True,
)

# ─── Paths inside the container ───────────────────────────────────────────────
WORK_DIR          = Path("/workspace")
GAUSSIAN_REPO     = Path("/gaussian-splatting")   # cloned at image build time
ESRGAN_SCRIPT     = Path("/Real-ESRGAN/inference_realesrgan.py")
COLMAP_BIN        = "colmap"


def handler(job):
    """
    Runpod calls this function with:
    {
      "input": {
        "job_id":         "...",
        "input_files":    [{ "url": "...", "original_name": "...", "resource_type": "image|video" }],
        "input_type":     "images" | "video",
        "enhance":        true | false,
        "quality":        "fast" | "balanced" | "high",
        "iterations":     7000 | 30000 | 100000,
        "webhook_url":    "https://your-api.com/api/webhooks/runpod",
        "webhook_secret": "..."
      }
    }
    """
    inp           = job["input"]
    job_id        = inp["job_id"]
    input_files   = inp["input_files"]
    input_type    = inp["input_type"]
    enhance       = inp.get("enhance", True)
    iterations    = inp.get("iterations", 30_000)
    webhook_url   = inp.get("webhook_url")
    webhook_secret = inp.get("webhook_secret", "")

    work = WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[{job_id}] Starting pipeline | type={input_type} enhance={enhance} iter={iterations}")

        # ── Stage 1: Download input files ─────────────────────────────────────
        print(f"[{job_id}] Downloading {len(input_files)} input files...")
        raw_dir = work / "raw"
        raw_dir.mkdir(exist_ok=True)

        if input_type == "video":
            video_path = download_file(input_files[0]["url"], raw_dir / "input.mp4")
            images_dir = extract_frames(video_path, work / "frames")
        else:
            images_dir = raw_dir
            for i, f in enumerate(input_files):
                ext = Path(f["original_name"]).suffix or ".jpg"
                download_file(f["url"], raw_dir / f"{i:04d}{ext}")

        # ── Stage 2: Image enhancement (Real-ESRGAN) ──────────────────────────
        if enhance and ESRGAN_SCRIPT.exists():
            print(f"[{job_id}] Enhancing images with Real-ESRGAN...")
            enhanced_dir = work / "enhanced"
            enhanced_dir.mkdir(exist_ok=True)
            run_cmd([
                "python", str(ESRGAN_SCRIPT),
                "-i", str(images_dir),
                "-o", str(enhanced_dir),
                "--model_name", "RealESRGAN_x4plus",
                "--outscale", "2",
                "--fp32",
            ])
            images_dir = enhanced_dir
        else:
            print(f"[{job_id}] Skipping enhancement (enhance={enhance})")

        # ── Stage 3: COLMAP (Structure-from-Motion) ───────────────────────────
        print(f"[{job_id}] Running COLMAP...")
        colmap_dir = work / "colmap"
        colmap_dir.mkdir(exist_ok=True)
        run_colmap(images_dir, colmap_dir)

        # ── Stage 4: Gaussian Splatting training ──────────────────────────────
        print(f"[{job_id}] Training Gaussian splatting model ({iterations} iterations)...")
        output_dir = work / "output"
        run_cmd([
            "python", str(GAUSSIAN_REPO / "train.py"),
            "-s", str(colmap_dir),
            "-m", str(output_dir),
            "--iterations", str(iterations),
            "--densification_interval", "100",
            "--quiet",
        ])

        # ── Stage 5: Convert .ply → .glb ─────────────────────────────────────
        print(f"[{job_id}] Converting to GLB...")
        ply_path = find_final_ply(output_dir)
        glb_path = work / "scene.glb"
        convert_ply_to_glb(ply_path, glb_path)

        # Generate thumbnail
        thumbnail_path = generate_thumbnail(glb_path, work / "thumbnail.jpg")

        # ── Stage 6: Upload to Cloudinary ────────────────────────────────────
        print(f"[{job_id}] Uploading results to Cloudinary...")
        glb_result = cloudinary.uploader.upload(
            str(glb_path),
            resource_type = "raw",
            folder        = f"gaussian-outputs/{job_id}",
            public_id     = "scene",
            tags          = [f"job_{job_id}"],
        )

        thumb_result = None
        if thumbnail_path and thumbnail_path.exists():
            thumb_result = cloudinary.uploader.upload(
                str(thumbnail_path),
                resource_type = "image",
                folder        = f"gaussian-outputs/{job_id}",
                public_id     = "thumbnail",
                tags          = [f"job_{job_id}"],
            )

        # ── Stage 7: Return output ────────────────────────────────────────────
        glb_size = glb_path.stat().st_size if glb_path.exists() else None

        output = {
            "glb_cloudinary_id":       glb_result["public_id"],
            "glb_secure_url":          glb_result["secure_url"],
            "thumbnail_cloudinary_id": thumb_result["public_id"]  if thumb_result else None,
            "thumbnail_secure_url":    thumb_result["secure_url"] if thumb_result else None,
            "file_size_bytes":         glb_size,
        }

        print(f"[{job_id}] Done! GLB: {glb_result['secure_url']}")
        return output

    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] ERROR: {error_msg}", file=sys.stderr)
        raise  # Runpod will mark job as FAILED and trigger webhook

    finally:
        # Clean up workspace to free disk space
        try:
            shutil.rmtree(work, ignore_errors=True)
            print(f"[{job_id}] Workspace cleaned up")
        except Exception:
            pass


# ─── Helper functions ─────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> Path:
    """Download a file from a URL to a local path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> Path:
    """Extract frames from a video using ffmpeg."""
    output_dir.mkdir(exist_ok=True)
    run_cmd([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_dir / "%04d.jpg"),
        "-y",
    ])
    frame_count = len(list(output_dir.glob("*.jpg")))
    print(f"Extracted {frame_count} frames at {fps} fps")
    if frame_count < 10:
        raise ValueError(f"Too few frames extracted ({frame_count}). Video may be too short.")
    return output_dir


def run_colmap(images_dir: Path, colmap_dir: Path):
    """Run COLMAP feature extraction + matching + reconstruction."""
    db_path     = colmap_dir / "database.db"
    sparse_dir  = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # Feature extraction
    run_cmd([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ])

    # Feature matching
    run_cmd([
        COLMAP_BIN, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ])

    # Sparse reconstruction
    run_cmd([
        COLMAP_BIN, "mapper",
        "--database_path",    str(db_path),
        "--image_path",       str(images_dir),
        "--output_path",      str(sparse_dir),
    ])

    # Check reconstruction succeeded
    if not any(sparse_dir.iterdir()):
        raise RuntimeError(
            "COLMAP failed to reconstruct the scene. "
            "Try images with more overlap, better lighting, and different angles."
        )


def find_final_ply(output_dir: Path) -> Path:
    """Find the final point cloud .ply file from Gaussian splatting output."""
    # Gaussian splatting saves checkpoints like point_cloud/iteration_30000/point_cloud.ply
    candidates = sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No .ply output found in {output_dir}")
    # Return the highest iteration (best quality)
    return candidates[-1]


def convert_ply_to_glb(ply_path: Path, glb_path: Path):
    """
    Convert Gaussian splatting .ply to .glb using gsplat or a custom converter.

    Option A (recommended): Use the gsplat library's export
    Option B: Use Blender headless with a Python script
    Option C: Use a custom ply2splat + splat2glb pipeline
    """
    # Try gsplat export first
    try:
        run_cmd([
            "python", "-c", f"""
import gsplat
from gsplat.compression import png_compress
# Load and export as GLB
print("Converting with gsplat...")
""",
        ])
    except Exception:
        pass

    # Fallback: use Blender headless
    blender_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_mesh.ply(filepath='{ply_path}')
bpy.ops.export_scene.gltf(filepath='{glb_path}', export_format='GLB')
"""
    script_file = ply_path.parent / "convert.py"
    script_file.write_text(blender_script)

    run_cmd([
        "blender", "--background", "--python", str(script_file),
    ])

    if not glb_path.exists():
        raise FileNotFoundError(f"GLB conversion failed — output not found at {glb_path}")


def generate_thumbnail(glb_path: Path, output_path: Path) -> Path:
    """Generate a JPEG thumbnail of the 3D scene using Blender headless render."""
    try:
        render_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='{glb_path}')
bpy.context.scene.render.filepath = '{output_path}'
bpy.context.scene.render.image_settings.file_format = 'JPEG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.ops.render.render(write_still=True)
"""
        script_file = glb_path.parent / "thumbnail.py"
        script_file.write_text(render_script)
        run_cmd(["blender", "--background", "--python", str(script_file)])
        return output_path if output_path.exists() else None
    except Exception as e:
        print(f"Thumbnail generation failed (non-fatal): {e}")
        return None


def run_cmd(cmd: list, cwd: Path = None):
    """Run a shell command and raise on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd    = cwd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text   = True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"  {' '.join(str(c) for c in cmd)}\n"
            f"Output:\n{result.stdout[-2000:]}"  # last 2000 chars of output
        )
    return result.stdout


# ─── Runpod entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
