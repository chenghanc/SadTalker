import cv2, os
import numpy as np
from tqdm import tqdm
import uuid

from src.utils.videoio import save_video_with_watermark 

def _first_frame(path: str):
    """Load the first frame from an image or video path."""
    if not os.path.isfile(path):
        raise ValueError(f'{path} must be a valid path to video/image file')
    if path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f'Failed to read image at {path}')
        return frame

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f'{path} must be a valid path to video/image file')
    still_reading, frame = cap.read()
    cap.release()
    if not still_reading:
        raise ValueError(f'No frames read from {path}')
    return frame

def _clamp_location(x, y, w, h, frame_w, frame_h):
    """Keep the seamlessClone center inside the frame, considering patch size."""
    half_w = max(w // 2, 0)
    half_h = max(h // 2, 0)
    x = max(min(x, frame_w - 1 - half_w), half_w)
    y = max(min(y, frame_h - 1 - half_h), half_h)
    return x, y

def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False, background_video=None, background_position=None, background_scale=1.0):
    background_stream = None
    static_full_img = None
    first_background = None

    # source size for keeping the original appearance scale
    source_first_frame = _first_frame(pic_path)
    source_h, source_w = source_first_frame.shape[0], source_first_frame.shape[1]

    if background_video:
        if not os.path.isfile(background_video):
            raise ValueError('background_video must be a valid path to video file')

        background_stream = cv2.VideoCapture(background_video)
        if not background_stream.isOpened():
            raise ValueError('background_video must be a valid path to video file')

        still_reading, first_background = background_stream.read()
        if not still_reading:
            background_stream.release()
            raise ValueError('No frames read from background_video')

        frame_h, frame_w = first_background.shape[0], first_background.shape[1]
    else:
        full_img = source_first_frame
        frame_h = full_img.shape[0]
        frame_w = full_img.shape[1]
        static_full_img = full_img

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    
    if len(crop_info) != 3:
        print("you didn't crop the image")
        if background_stream:
            background_stream.release()
        return
    else:
        _, _ = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    # keep the generated patch scaled to match the original source size (clamped to background frame)
    resize_scale = min(frame_w / source_w, frame_h / source_h, 1.0)
    scale = max(background_scale, 0.01)
    target_w = min(frame_w, max(1, int(source_w * resize_scale * scale)))
    target_h = min(frame_h, max(1, int(source_h * resize_scale * scale)))

    tmp_path = str(uuid.uuid4())+'.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))

    current_bg = first_background
    while True:
        still_reading, crop_frame = video_stream.read()
        if not still_reading:
            break

        if background_stream is not None:
            if current_bg is None:
                bg_reading, current_bg = background_stream.read()
                if not bg_reading:
                    # restart the background stream when it runs out to avoid storing all frames in memory
                    background_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    bg_reading, current_bg = background_stream.read()
                    if not bg_reading:
                        background_stream.release()
                        video_stream.release()
                        out_tmp.release()
                        raise ValueError('No frames read from background_video')
            target_bg = current_bg
            current_bg = None
            # place the subject based on provided normalized position, otherwise center on background video
            if background_position and len(background_position) == 2:
                px = min(max(background_position[0], 0.0), 1.0)
                py = min(max(background_position[1], 0.0), 1.0)
                location = (int(frame_w * px), int(frame_h * py))
            else:
                location = (frame_w // 2, frame_h // 2)
        else:
            target_bg = static_full_img
            if background_position and len(background_position) == 2:
                px = min(max(background_position[0], 0.0), 1.0)
                py = min(max(background_position[1], 0.0), 1.0)
                location = (int(frame_w * px), int(frame_h * py))
            else:
                location = ((ox1+ox2) // 2, (oy1+oy2) // 2)

        location = _clamp_location(location[0], location[1], target_w, target_h, frame_w, frame_h)

        p = cv2.resize(crop_frame.astype(np.uint8), (target_w, target_h)) 
        mask = 255*np.ones(p.shape, p.dtype)

        gen_img = cv2.seamlessClone(p, target_bg, mask, location, cv2.NORMAL_CLONE)
        out_tmp.write(gen_img)

    video_stream.release()
    if background_stream:
        background_stream.release()
    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)
