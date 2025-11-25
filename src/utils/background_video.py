import os
import uuid

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.videoio import save_video_with_watermark


def _resolve_target_dimensions(crop_info, preprocess):
    if not crop_info or len(crop_info) != 3:
        return None, None

    try:
        _, (clx, cly, crx, cry), (lx, ly, rx, ry) = crop_info
    except (ValueError, TypeError):
        return None, None

    extended_crop = isinstance(preprocess, str) and 'ext' in preprocess.lower()
    if extended_crop:
        oy1, oy2, ox1, ox2 = cly, cry, clx, crx
    else:
        oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx

    width = int(abs(ox2 - ox1))
    height = int(abs(oy2 - oy1))
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _extract_audio(source_video, audio_path):
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{source_video}" -vn -acodec pcm_s16le -ar 16000 "{audio_path}"'
    status = os.system(cmd)
    if status != 0 or not os.path.isfile(audio_path):
        raise RuntimeError('Failed to extract audio from generated video for background compositing')


def _next_background_frame(cap, fallback_frame):
    ok, frame = cap.read()
    if ok:
        return frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if ok:
        return frame

    # if the capture still fails (e.g. webcam stream), reuse fallback
    return fallback_frame.copy()


def composite_with_background_video(foreground_video_path, background_video_path, crop_info, save_dir, preprocess='crop'):
    if not os.path.isfile(foreground_video_path):
        raise ValueError('Foreground video was not found. Cannot composite with background video.')
    if not os.path.isfile(background_video_path):
        raise ValueError('Background video path is invalid or does not exist.')

    os.makedirs(save_dir, exist_ok=True)

    fg_cap = cv2.VideoCapture(foreground_video_path)
    if not fg_cap.isOpened():
        raise ValueError('Failed to open generated foreground video for background compositing.')

    still_reading, current_fg = fg_cap.read()
    if not still_reading:
        fg_cap.release()
        raise ValueError('No frames were found inside the generated video. Cannot composite background.')

    fg_fps = fg_cap.get(cv2.CAP_PROP_FPS)
    if fg_fps <= 0:
        fg_fps = 25
    fg_total_frames = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg_cap = cv2.VideoCapture(background_video_path)
    if not bg_cap.isOpened():
        fg_cap.release()
        raise ValueError('Failed to open background video for compositing.')

    ok, sample_bg = bg_cap.read()
    if not ok:
        fg_cap.release()
        bg_cap.release()
        raise ValueError('Background video does not contain readable frames.')
    bg_height, bg_width = sample_bg.shape[:2]
    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    target_width, target_height = _resolve_target_dimensions(crop_info, preprocess)
    if target_width is None or target_height is None:
        target_height, target_width = current_fg.shape[0], current_fg.shape[1]

    scale = min(bg_width / max(target_width, 1), bg_height / max(target_height, 1), 1.0)
    scaled_width = max(1, int(round(target_width * scale)))
    scaled_height = max(1, int(round(target_height * scale)))

    tmp_composite_path = os.path.join(save_dir, f'{uuid.uuid4().hex}_bg_tmp.mp4')
    video_writer = cv2.VideoWriter(
        tmp_composite_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fg_fps,
        (bg_width, bg_height)
    )

    progress = tqdm(total=fg_total_frames if fg_total_frames > 0 else None,
                    desc='background_composite', unit='frame', leave=False)

    try:
        while current_fg is not None:
            bg_frame = _next_background_frame(bg_cap, sample_bg)
            resized_foreground = cv2.resize(current_fg, (scaled_width, scaled_height))
            mask = 255 * np.ones(resized_foreground.shape, dtype=resized_foreground.dtype)
            center_point = (bg_width // 2, bg_height // 2)
            composed = cv2.seamlessClone(resized_foreground, bg_frame, mask, center_point, cv2.NORMAL_CLONE)
            video_writer.write(composed)

            if progress is not None:
                progress.update(1)

            still_reading, next_fg = fg_cap.read()
            if not still_reading:
                current_fg = None
            else:
                current_fg = next_fg
    finally:
        video_writer.release()
        fg_cap.release()
        bg_cap.release()
        if progress is not None:
            progress.close()

    tmp_audio_path = os.path.join(save_dir, f'{uuid.uuid4().hex}_bg_audio.wav')
    _extract_audio(foreground_video_path, tmp_audio_path)

    base_name = os.path.splitext(os.path.basename(foreground_video_path))[0]
    final_path = os.path.join(save_dir, f'{base_name}_with_background.mp4')
    save_video_with_watermark(tmp_composite_path, tmp_audio_path, final_path, watermark=False)

    os.remove(tmp_composite_path)
    os.remove(tmp_audio_path)

    return final_path
