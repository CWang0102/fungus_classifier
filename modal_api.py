import modal

# ---------------------------------------------------------------------------
# Modal Image – install all dependencies into the container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "fastapi[standard]",
        "pillow",
        "numpy",
        "opencv-python-headless",
        "python-multipart",
    )
    # Copy your local model.py into the container so imports work
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("./model/fine_tuned.pth", "/root/fine_tuned.pth")
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("fungus-classification-api", image=image)

# ---------------------------------------------------------------------------
# Constants (same as your original)
# ---------------------------------------------------------------------------
NUM_SHAPE = 3
NUM_EDGE = 2
NUM_TEXTURE = 2
NUM_SIZE = 2
NUM_COLOUR = 2

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 20
MAX_ARCHIVE_SIZE_BYTES = 200 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_DIMENSION = 1024  # Maximum dimension for input images

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
CLASS_NAME = (
    "不规则形、平滑状、绒毛状、大、白色",
    "不规则形、平滑状、绒毛状、小、白色",
    "不规则形、平滑状、绒毛状、小、褐色",
    "不规则形、毛刺状、绒毛状、小、白色",
    "圆形、平滑状、粘液状且绒毛状、大、白色",
    "圆形、平滑状、绒毛状、大、白色",
    "圆形、平滑状、绒毛状、大、褐色",
    "圆形、平滑状、绒毛状、小、白色",
    "圆形、毛刺状、粘液状且绒毛状、大、白色",
    "圆形、毛刺状、绒毛状、大、白色",
    "圆形、毛刺状、绒毛状、小、白色",
    "环形、平滑状、绒毛状、大、白色",
    "环形、毛刺状、绒毛状、大、白色",
)


# ---------------------------------------------------------------------------
# FastAPI application (built inside a function so imports happen in-container)
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",
    timeout=300,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    """Return a fully-configured FastAPI instance."""

    import sys, os, shutil, zipfile, tarfile, base64, time
    from pathlib import Path
    from uuid import uuid4
    from datetime import datetime, timezone
    from io import BytesIO
    from typing import List

    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image
    from fastapi import FastAPI, File, UploadFile, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from uuid import UUID

    # Make model.py importable
    sys.path.insert(0, "/root")
    from model import Resnet18, ResNet18_Weights, stats_to_class

    # ---- Pydantic models ----
    class ImageAnalysis(BaseModel):
        filename: str
        predicted_class: str
        image_data: str
        cam_data: str

    class AnalysisResponse(BaseModel):
        upload_id: UUID
        upload_time: datetime
        analysis: List[ImageAnalysis]
        processing_time: float

    # ---- Load model ONCE at container start (runs before first request) ----
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {DEVICE} …")

    MODEL = Resnet18(NUM_SHAPE, NUM_EDGE, NUM_TEXTURE, NUM_SIZE, NUM_COLOUR)
    weights_path = Path("/root/fine_tuned.pth")
    MODEL.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    TRANSFORM = ResNet18_Weights.IMAGENET1K_V1.transforms()
    print("Model loaded ✓")

    # Use /tmp for ephemeral uploads (serverless-safe)
    UPLOAD_DIR = Path("/tmp/uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)

    # ---- Helper functions ----
    def sanitize_filename(filename: str) -> str:
        return Path(filename).name

    def validate_upload_file(file: UploadFile, max_size: int):
        if not file.filename:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported: {suffix}")
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Invalid MIME type")
        if file.size is not None and file.size > max_size:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    def validate_image_content(path: Path):
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.width * img.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Resolution too large")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or corrupted image")

    def safe_extract_zip(zip_ref, extract_to: Path):
        for member in zip_ref.infolist():
            extracted_path = extract_to / member.filename
            if not extracted_path.resolve().is_relative_to(extract_to.resolve()):
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Archive contains unsafe paths")
        zip_ref.extractall(extract_to)

    def analyse_image(image_path: Path) -> ImageAnalysis:
        try:
            # Load and resize image if needed
            image_pil = Image.open(image_path).convert("RGB")
            
            # Quick win #2: Resize large images
            if max(image_pil.size) > MAX_DIMENSION:
                image_pil.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            
            image_array = np.array(image_pil)
            image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)
            image_tensor.requires_grad = True

            activations, gradients = [], []

            def forward_hook(module, inp, out):
                activations.append(out)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            layer = MODEL.backbone.layer4[-1].conv2
            fh = layer.register_forward_hook(forward_hook)
            bh = layer.register_full_backward_hook(backward_hook)

            try:
                shape_pred, edge_pred, texture_pred, size_pred, colour_pred = MODEL(image_tensor)

                _, shape_hat = shape_pred.max(1)
                _, edge_hat = edge_pred.max(1)
                _, texture_hat = texture_pred.max(1)
                _, size_hat = size_pred.max(1)
                _, colour_hat = colour_pred.max(1)

                key = (
                    int(shape_hat[0].item()),
                    int(edge_hat[0].item()),
                    int(texture_hat[0].item()),
                    int(size_hat[0].item()),
                    int(colour_hat[0].item()),
                )
                pred_class = stats_to_class.get(key, "Unknown")

                # Grad-CAM
                one_hot = torch.zeros_like(shape_pred)
                one_hot[0][shape_hat[0]] = 1.0
                MODEL.zero_grad()
                shape_pred.backward(gradient=one_hot, retain_graph=False)

                act = activations[0][0]
                grad = gradients[0][0]
                weights = torch.mean(grad, dim=(1, 2))
                cam = torch.zeros(act.shape[1:], device=DEVICE)
                for i, w_val in enumerate(weights):
                    cam += w_val * act[i]
                cam = F.relu(cam)
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                cam_np = cam.detach().cpu().numpy()
            finally:
                fh.remove()
                bh.remove()

            # Overlay creation
            h, w = image_array.shape[:2]
            cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(image_array, 0.5, heatmap, 0.5, 0)

            # Quick win #1: Reduced JPEG quality and added optimize flag
            buf_orig = BytesIO()
            image_pil.save(buf_orig, format="JPEG", quality=65, optimize=True)
            buf_orig.seek(0)
            image_data = f"data:image/jpeg;base64,{base64.b64encode(buf_orig.read()).decode()}"

            buf_cam = BytesIO()
            Image.fromarray(overlay).save(buf_cam, format="JPEG", quality=65, optimize=True)
            buf_cam.seek(0)
            cam_data = f"data:image/jpeg;base64,{base64.b64encode(buf_cam.read()).decode()}"

            if DEVICE == "cuda":
                del image_tensor, act, grad, cam
                torch.cuda.empty_cache()

            return ImageAnalysis(
                filename=image_path.name,
                predicted_class=CLASS_NAME[pred_class],
                image_data=image_data,
                cam_data=cam_data,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Error processing image: {e}")

    # ---- Build FastAPI ----
    web_app = FastAPI(
        title="Fungus Image Classification",
        description="Upload a fungus image and receive classification with Grad-CAM visualization.",
    )
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/", tags=["Root"])
    async def root():
        return {
            "status": "Online",
            "message": "Fungus image analysis API with CAM visualization (Modal)",
            "device": DEVICE,
        }

    @web_app.post("/upload", response_model=AnalysisResponse, tags=["Upload"])
    async def upload_image(file: UploadFile = File(...)):
        start = datetime.now(timezone.utc)
        validate_upload_file(file, MAX_IMAGE_SIZE_BYTES)
        uid = uuid4()
        fpath = UPLOAD_DIR / f"{uid}_{sanitize_filename(file.filename)}"
        try:
            contents = await file.read()
            if len(contents) > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
            fpath.write_bytes(contents)
            validate_image_content(fpath)
            result = analyse_image(fpath)
            return AnalysisResponse(
                upload_id=uid,
                upload_time=start,
                analysis=[result],
                processing_time=(datetime.now(timezone.utc) - start).total_seconds(),
            )
        finally:
            if fpath.exists():
                fpath.unlink()

    @web_app.post("/upload/batch", response_model=AnalysisResponse, tags=["Upload"])
    async def upload_batch(files: List[UploadFile] = File(...)):
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Too many files")
        start = datetime.now(timezone.utc)
        batch_id = uuid4()
        results, created = [], []
        try:
            for idx, f in enumerate(files):
                validate_upload_file(f, MAX_IMAGE_SIZE_BYTES)
                fpath = UPLOAD_DIR / f"{batch_id}_{idx}_{sanitize_filename(f.filename)}"
                created.append(fpath)
                fpath.write_bytes(await f.read())
                validate_image_content(fpath)
                results.append(analyse_image(fpath))
            return AnalysisResponse(
                upload_id=batch_id,
                upload_time=start,
                analysis=results,
                processing_time=(datetime.now(timezone.utc) - start).total_seconds(),
            )
        finally:
            for p in created:
                if p.exists():
                    p.unlink()

    @web_app.post("/upload/compressed", response_model=AnalysisResponse, tags=["Upload"])
    async def upload_compressed(file: UploadFile = File(...)):
        if file.size is not None and file.size > MAX_ARCHIVE_SIZE_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Archive too large")
        start = datetime.now(timezone.utc)
        uid = uuid4()
        safe_name = sanitize_filename(file.filename)
        archive_path = UPLOAD_DIR / f"{uid}_{safe_name}"
        extract_dir = UPLOAD_DIR / f"{uid}_extracted"
        try:
            archive_path.write_bytes(await file.read())
            extract_dir.mkdir(exist_ok=True)
            if safe_name.lower().endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    safe_extract_zip(zf, extract_dir)
            elif safe_name.lower().endswith((".tar", ".tar.gz", ".tgz")):
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(extract_dir)
            else:
                raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported archive format")
            images = [f for f in extract_dir.rglob("*") if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
            if not images:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "No valid images in archive")
            results, errors = [], []
            for img_path in images:
                validate_image_content(img_path)
                try:
                    results.append(analyse_image(img_path))
                except Exception as e:
                    errors.append(f"{img_path.name}: {e}")
            if not results:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"All failed: {'; '.join(errors)}")
            return AnalysisResponse(
                upload_id=uid,
                upload_time=start,
                analysis=results,
                processing_time=(datetime.now(timezone.utc) - start).total_seconds(),
            )
        finally:
            if archive_path.exists():
                archive_path.unlink()
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

    @web_app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "device": DEVICE, "model_loaded": MODEL is not None}

    return web_app