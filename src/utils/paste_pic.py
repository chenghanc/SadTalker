import cv2, os
import numpy as np
from tqdm import tqdm
import uuid

from src.utils.videoio import save_video_with_watermark 

def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False, background_video=None):
    background_stream = None
    static_full_img = None
    first_background = None

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
        if not os.path.isfile(pic_path):
            raise ValueError('pic_path must be a valid path to video/image file')
        elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_img = cv2.imread(pic_path)
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(pic_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                video_stream.release()
                break 
            full_img = frame
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
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    tmp_path = str(uuid.uuid4())+'.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))

    current_bg = first_background
    for crop_frame in tqdm(iter(lambda: video_stream.read(), (False, None)), 'seamlessClone:'):
        still_reading, crop_frame = crop_frame
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
        else:
            target_bg = static_full_img

        p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1)) 
        mask = 255*np.ones(p.shape, p.dtype)
        location = ((ox1+ox2) // 2, (oy1+oy2) // 2)

        gen_img = cv2.seamlessClone(p, target_bg, mask, location, cv2.NORMAL_CLONE)
        out_tmp.write(gen_img)

    video_stream.release()
    if background_stream:
        background_stream.release()
    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)
