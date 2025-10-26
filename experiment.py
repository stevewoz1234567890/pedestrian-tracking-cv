"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    import cv2
    import numpy as np
    
    # Load images
    imgs_dir = os.path.join(input_dir, "TUD-Campus")
    imgs_list = [f for f in os.listdir(imgs_dir) if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()
    
    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Initialize multiple Kalman filters for different targets
    kf_list = []
    target_templates = []
    target_centers = []
    
    save_frames = {
        28: os.path.join(output_dir, 'ps5-5-a-1.png'),  # Frame 29 (0-indexed)
        55: os.path.join(output_dir, 'ps5-5-a-2.png'),  # Frame 56 (0-indexed)
        70: os.path.join(output_dir, 'ps5-5-a-3.png')   # Frame 71 (0-indexed)
    }
    
    frame_num = 0
    
    for img in imgs_list:
        frame = cv2.imread(os.path.join(imgs_dir, img))
        
        # Detect people using HOG
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                              padding=(8, 8), scale=1.05)
        
        # Initialize Kalman filters for new detections
        if frame_num == 0 and len(rects) > 0:
            # Initialize with top detections
            for i, (x, y, w, h) in enumerate(rects[:3]):  # Track up to 3 people
                center_x = x + w // 2
                center_y = y + h // 2
                kf = ps5.KalmanFilter(center_x, center_y)
                kf_list.append(kf)
                target_centers.append((center_x, center_y))
                target_templates.append(frame[y:y+h, x:x+w])
        
        # Update existing trackers
        for i, kf in enumerate(kf_list):
            if i < len(rects):
                x, y, w, h = rects[i]
                center_x = x + w // 2
                center_y = y + h // 2
                x_pred, y_pred = kf.process(center_x, center_y)
                target_centers[i] = (int(x_pred), int(y_pred))
        
        # Render and save output frames
        if frame_num in save_frames:
            frame_out = frame.copy()
            
            # Draw tracking boxes for each target
            for i, (center_x, center_y) in enumerate(target_centers):
                if i < len(target_templates):
                    h, w = target_templates[i].shape[:2]
                    cv2.rectangle(frame_out, 
                                  (center_x - w//2, center_y - h//2),
                                  (center_x + w//2, center_y + h//2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame_out, f'Target {i+1}', 
                              (center_x - w//2, center_y - h//2 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(save_frames[frame_num], frame_out)
        
        frame_num += 1
        if frame_num % 10 == 0:
            print(f'Working on frame {frame_num}')


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    import cv2
    import numpy as np
    
    # Load images
    imgs_dir = os.path.join(input_dir, "follow")
    imgs_list = [f for f in os.listdir(imgs_dir) if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()
    
    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Initialize Kalman filter for the man with hat
    kf = None
    target_center = None
    target_template = None
    
    save_frames = {
        60: os.path.join(output_dir, 'ps5-6-a-1.png'),
        160: os.path.join(output_dir, 'ps5-6-a-2.png'),
        186: os.path.join(output_dir, 'ps5-6-a-3.png')
    }
    
    frame_num = 0
    
    for img in imgs_list:
        frame = cv2.imread(os.path.join(imgs_dir, img))
        
        # Detect people using HOG
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                              padding=(8, 8), scale=1.05)
        
        # Initialize tracker on first frame
        if frame_num == 0 and len(rects) > 0:
            # Select the largest detection (likely the man with hat)
            largest_idx = np.argmax([w*h for x, y, w, h in rects])
            x, y, w, h = rects[largest_idx]
            center_x = x + w // 2
            center_y = y + h // 2
            kf = ps5.KalmanFilter(center_x, center_y)
            target_center = (center_x, center_y)
            target_template = frame[y:y+h, x:x+w]
        
        # Update tracker
        if kf is not None and len(rects) > 0:
            # Find the detection closest to current target position
            if target_center is not None:
                current_x, current_y = target_center
                distances = [np.sqrt((x + w//2 - current_x)**2 + (y + h//2 - current_y)**2) 
                           for x, y, w, h in rects]
                closest_idx = np.argmin(distances)
                x, y, w, h = rects[closest_idx]
                center_x = x + w // 2
                center_y = y + h // 2
                x_pred, y_pred = kf.process(center_x, center_y)
                target_center = (int(x_pred), int(y_pred))
        
        # Render and save output frames
        if frame_num in save_frames and target_center is not None:
            frame_out = frame.copy()
            
            # Draw tracking box
            if target_template is not None:
                h, w = target_template.shape[:2]
                center_x, center_y = target_center
                cv2.rectangle(frame_out, 
                            (center_x - w//2, center_y - h//2),
                            (center_x + w//2, center_y + h//2),
                            (0, 255, 0), 2)
                cv2.putText(frame_out, 'Man with Hat', 
                          (center_x - w//2, center_y - h//2 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(save_frames[frame_num], frame_out)
        
        frame_num += 1
        if frame_num % 20 == 0:
            print(f'Working on frame {frame_num}')


if __name__ == '__main__':
    part_1b()
    part_1c()
    part_2a()
    part_2b()
    part_3()
    part_4()
    part_5()
    part_6()
