from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DOSSIERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "videos_originales"
OUTPUT_DIR = BASE_DIR / "videos_augmentees"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# PARAMÈTRES
# ============================================================

# Éclairage très fort mais encore visible
BRIGHT_ALPHA = 1.65
BRIGHT_BETA = 65

# Accélération très visible
SPEED_FACTOR = 2.0

# Dégradation forte
NOISE_STD = 34
JPEG_QUALITY = 25
DOWNSCALE_FACTOR = 0.55


# ============================================================
# TRANSFORMATIONS
# ============================================================

def horizontal_flip(frame: np.ndarray) -> np.ndarray:
    """
    Retourne horizontalement toute la frame.
    """
    return cv2.flip(frame, 1)


def very_bright(frame: np.ndarray) -> np.ndarray:
    """
    Rend la vidéo beaucoup plus lumineuse,
    avec un contraste nettement renforcé.
    """
    return cv2.convertScaleAbs(
        frame,
        alpha=BRIGHT_ALPHA,
        beta=BRIGHT_BETA,
    )


def strong_degradation(frame: np.ndarray) -> np.ndarray:
    """
    Dégradation forte de type ancienne caméra de surveillance :
    - perte de résolution ;
    - bruit/grésillement ;
    - léger flou ;
    - lignes horizontales ;
    - forte compression JPEG.
    """

    height, width = frame.shape[:2]

    # 1. Réduire fortement la résolution
    small_width = max(1, int(width * DOWNSCALE_FACTOR))
    small_height = max(1, int(height * DOWNSCALE_FACTOR))

    degraded = cv2.resize(
        frame,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )

    # 2. Revenir à la résolution originale
    degraded = cv2.resize(
        degraded,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # 3. Léger flou
    degraded = cv2.GaussianBlur(
        degraded,
        (5, 5),
        0,
    )

    # 4. Bruit gaussien fort
    noise = np.random.normal(
        loc=0,
        scale=NOISE_STD,
        size=degraded.shape,
    ).astype(np.float32)

    degraded = degraded.astype(np.float32) + noise
    degraded = np.clip(
        degraded,
        0,
        255,
    ).astype(np.uint8)

    # 5. Pixels noirs et blancs aléatoires
    salt_mask = (
        np.random.random((height, width))
        < 0.012
    )
    pepper_mask = (
        np.random.random((height, width))
        < 0.008
    )

    degraded[salt_mask] = 255
    degraded[pepper_mask] = 0

    # 6. Lignes horizontales type grésillement
    for y in range(0, height, 10):
        degraded[y:y + 1] = (
            degraded[y:y + 1].astype(np.float32)
            * 0.65
        ).astype(np.uint8)

    # 7. Compression JPEG très visible
    success, encoded = cv2.imencode(
        ".jpg",
        degraded,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY,
        ],
    )

    if success:
        degraded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

    return degraded


def identity(frame: np.ndarray) -> np.ndarray:
    """
    Ne modifie pas l'image.
    Utilisé pour la variante accélérée.
    """
    return frame


# ============================================================
# TRAITEMENT D'UNE VIDÉO
# ============================================================

def transform_video(
    input_path: Path,
    output_path: Path,
    transform_function,
    fps_factor: float = 1.0,
) -> None:

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir : {input_path}"
        )

    original_fps = capture.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 25.0

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_fps = original_fps * fps_factor

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Impossible de créer : {output_path}"
        )

    written_frames = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        transformed_frame = transform_function(frame)

        writer.write(transformed_frame)
        written_frames += 1

    capture.release()
    writer.release()

    print(
        f"Créée : {output_path.name} "
        f"| {written_frames} frames "
        f"| {output_fps:.2f} FPS"
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        raise FileNotFoundError(
            f"Aucune vidéo trouvée dans : {INPUT_DIR}"
        )

    print(
        f"Vidéos originales trouvées : "
        f"{len(video_paths)}\n"
    )

    for index, input_path in enumerate(
        video_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(video_paths)}] "
            f"{input_path.name}"
        )

        base_name = input_path.stem

        # 1. Retournement horizontal
        transform_video(
            input_path=input_path,
            output_path=OUTPUT_DIR / (
                f"{base_name}__aug_flip.mp4"
            ),
            transform_function=horizontal_flip,
        )

        # 2. Éclairage très fort
        transform_video(
            input_path=input_path,
            output_path=OUTPUT_DIR / (
                f"{base_name}__aug_bright.mp4"
            ),
            transform_function=very_bright,
        )

        # 3. Dégradation forte
        transform_video(
            input_path=input_path,
            output_path=OUTPUT_DIR / (
                f"{base_name}__aug_degraded.mp4"
            ),
            transform_function=strong_degradation,
        )

        # 4. Accélération x2
        transform_video(
            input_path=input_path,
            output_path=OUTPUT_DIR / (
                f"{base_name}__aug_speed_x2.mp4"
            ),
            transform_function=identity,
            fps_factor=SPEED_FACTOR,
        )

        print()

    generated_videos = list(
        OUTPUT_DIR.glob("*.mp4")
    )

    expected_count = len(video_paths) * 4

    print("========================================")
    print("AUGMENTATION TERMINÉE")
    print("========================================")
    print(
        f"Vidéos originales : {len(video_paths)}"
    )
    print(
        f"Vidéos augmentées attendues : "
        f"{expected_count}"
    )
    print(
        f"Vidéos augmentées créées : "
        f"{len(generated_videos)}"
    )
    print(
        f"Total avec originales : "
        f"{len(video_paths) + len(generated_videos)}"
    )

    if len(generated_videos) != expected_count:
        print(
            "Attention : le nombre de fichiers "
            "ne correspond pas au nombre attendu."
        )


if __name__ == "__main__":
    main()