import cv2
import numpy as np
from pathlib import Path
import os
import random
from fer import FER 

happyCol = [102,255,255]
sadCol = [255, 51, 51]
angryCol = [51, 51, 255]
surpriseCol = [255, 0, 255]

prevCols = ['angry', 'sad', 'happy']


def create_dynamic_comic_video(colors, input_path, output_path=None):
   darkCol, midCol, lightCol = colors
   print(midCol)
   try:
       if not os.path.exists(input_path):
           raise FileNotFoundError(f"Input video file not found: {input_path}")


       if output_path is None:
           input_file = Path(input_path)
           output_path = str(input_file.parent / f"{input_file.stem}_comic{input_file.suffix}")


       video = cv2.VideoCapture(input_path)
       if not video.isOpened():
           raise ValueError(f"Could not open video file: {input_path}")


       fps = video.get(cv2.CAP_PROP_FPS)
       width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
       total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


       face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       action_texts = ['SPLAT!', 'SWOOSH!', 'SMASH!', 'CRASH!', 'SLAM!', 'BAM!', 'WHACK!']


       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


       def create_action_bubble(text, size=300):
           bubble = np.zeros((size, size, 4), dtype=np.uint8)
           center = (size//2, size//2)
           points = []
           num_points = 12
           for i in range(num_points * 2):
               angle = i * (2 * np.pi / (num_points * 2))
               radius = size//2 if i % 2 == 0 else size//3
               x_point = int(center[0] + radius * np.cos(angle))
               y_point = int(center[1] + radius * np.sin(angle))
               points.append([x_point, y_point])
          
           points = np.array(points, np.int32)
           cv2.fillPoly(bubble, [points], (255, 255, 0, 255))
          
           font = cv2.FONT_HERSHEY_DUPLEX
           font_scale = 3.0
           thickness = 4
           text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
           text_x = (size - text_size[0]) // 2
           text_y = (size + text_size[1]) // 2
          
           cv2.putText(bubble, text, (text_x, text_y), font, font_scale, (0, 0, 0, 255), thickness + 3)
           cv2.putText(bubble, text, (text_x, text_y), font, font_scale, (255, 255, 255, 255), thickness)
          
           return bubble


       def detect_motion(frame1, frame2, threshold=0.2):
           if frame1 is None or frame2 is None:
               return False, None


           gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
           gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


           # Compute absolute difference between frames
           diff = cv2.absdiff(gray1, gray2)
           _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
           motion_mask = cv2.dilate(thresh, None, iterations=2)


           # Find contours of moving objects
           contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           if contours:
               # Find the largest contour (assuming it's the motion)
               largest_contour = max(contours, key=cv2.contourArea)
               if cv2.contourArea(largest_contour) > 5000:
                   x, y, w, h = cv2.boundingRect(largest_contour)
                   center_x = x + w // 2
                   center_y = y + h // 2
                   return True, (center_x, center_y, w, h)
           return False, None


       def blend_bubble(frame, bubble, position):
           x, y = position
           bubble_h, bubble_w = bubble.shape[:2]
          
           y1 = max(0, y - bubble_h//2)
           y2 = min(frame.shape[0], y + bubble_h//2)
           x1 = max(0, x - bubble_w//2)
           x2 = min(frame.shape[1], x + bubble_w//2)
          
           bubble_y1 = bubble_h//2 - (y2 - y1)//2
           bubble_y2 = bubble_y1 + (y2 - y1)
           bubble_x1 = bubble_w//2 - (x2 - x1)//2
           bubble_x2 = bubble_x1 + (x2 - x1)
          
           alpha = bubble[bubble_y1:bubble_y2, bubble_x1:bubble_x2, 3] / 255.0
           alpha = np.expand_dims(alpha, axis=-1)
          
           bubble_rgb = bubble[bubble_y1:bubble_y2, bubble_x1:bubble_x2, :3]
           frame_part = frame[y1:y2, x1:x2]
           blended = frame_part * (1 - alpha) + bubble_rgb * alpha
           frame[y1:y2, x1:x2] = blended.astype(np.uint8)
          
           return frame


       def balanced_comic_effect(frame):
           scale_factor = 1.5
           work_width = int(width / scale_factor)
           work_height = int(height / scale_factor)
           small_frame = cv2.resize(frame, (work_width, work_height))
          
           smooth = cv2.bilateralFilter(small_frame, 9, 150, 150)
           gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
          
           edges = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       9, 9)
          
           _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
           _, light = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
          
           edges = cv2.resize(edges, (width, height))
           dark = cv2.resize(dark, (width, height))
           light = cv2.resize(light, (width, height))
          
           comic = np.zeros_like(frame)
           comic[cv2.bitwise_not(dark) > 0] = darkCol
           comic[cv2.bitwise_xor(dark, light) > 0] = midCol
           comic[light > 0] = lightCol
          
           edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
           comic = cv2.bitwise_and(comic, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
          
           return cv2.convertScaleAbs(comic, alpha=1.1, beta=0)


       def find_safe_position(motion_center, face_regions, bubble_size, frame_shape):
           x, y, w, h = motion_center
           bubble_half = bubble_size // 2


           # Initial position is center of motion bounding box
           position_x = x
           position_y = y


           # Adjust position if overlapping with faces
           if not any(check_overlap((position_x, position_y), face_region, bubble_half)
                     for face_region in face_regions):
               return (position_x, position_y)
          
           # Try positions in expanding circles
           for r in range(50, 300, 50):
               for angle in range(0, 360, 45):
                   new_x = int(position_x + r * np.cos(np.radians(angle)))
                   new_y = int(position_y + r * np.sin(np.radians(angle)))
                  
                   if (0 + bubble_half < new_x < frame_shape[1] - bubble_half and
                       0 + bubble_half < new_y < frame_shape[0] - bubble_half and
                       not any(check_overlap((new_x, new_y), face_region, bubble_half)
                              for face_region in face_regions)):
                       return (new_x, new_y)
          
           return (position_x, position_y)


       def check_overlap(point, face_region, bubble_half):
           x, y = point
           x1, y1, x2, y2 = face_region
           return not (x + bubble_half < x1 or
                      x - bubble_half > x2 or
                      y + bubble_half < y1 or
                      y - bubble_half > y2)


       # Initialize variables
       prev_frame = None
       prev_comic_frame = None
       last_effect_frame = 0
       effect_cooldown = int(fps * 1)  # 1 second between effects
       min_frames_between_effects = int(fps * 0)  # Minimum 1.5 seconds between effects
       frame_count = 0
       effect_duration = int(fps * 0)  # 2 seconds duration
       effect_pause_frames = effect_duration  # Number of frames to pause
       current_effect = None
       effect_frames_remaining = 0


       # Store faces for smoother detection
       face_regions = []
       face_detection_interval = 5

       emotion_detector = FER(mtcnn=True)

       while True:
           ret, frame = video.read()
           if not ret:
               break
           
           if frame_count % 4 != 0:  # Skip odd-numbered frames (you can change to % 2 == 0 to skip even frames)
            out.write(prev_comic_frame)
            cv2.imshow("Comic Style", prev_comic_frame)
            frame_count += 1
            continue  # Skip this frame and move to the next one


           comic_frame = balanced_comic_effect(frame)

           emotions = emotion_detector.detect_emotions(frame)
           if emotions:
                dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                print(f"Detected emotion: {dominant_emotion}")

                prevCols.pop(0)
                prevCols.append(dominant_emotion)

                if(prevCols[0] == prevCols [1] == prevCols[2]):

                    # Modify cartoon effect based on detected emotion
                    if dominant_emotion == "happy":
                        color_boost = 40  # Increase saturation for happy
                        outline_strength = 7  # Softer outlines
                        
                        gradientColor(midCol, happyCol)

                    elif dominant_emotion == "sad":
                        color_boost = -30  # Reduce saturation for sad
                        outline_strength = 11  # Softer, less sharp outlines

                        gradientColor(midCol, sadCol)
                        
                    elif dominant_emotion == "angry":
                        color_boost = 20  # Warmer tones for anger
                        outline_strength = 3  # Sharper outlines
                        
                        gradientColor(midCol, angryCol)

                    elif dominant_emotion == "surprise":
                        color_boost = 50  # High saturation for surprise
                        outline_strength = 5

                        gradientColor(midCol, surpriseCol)
                else:
                    color_boost = 0  # Neutral
                    outline_strength = 9
                    darkCol, midCol, lightCol = colors
           else:
                # Default values if no emotion is detected
                color_boost = 0
                outline_strength = 9


           # Update face detection periodically
           if frame_count % face_detection_interval == 0:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               faces = face_cascade.detectMultiScale(gray, 1.1, 4)
               face_regions = [(x, y, x+w, y+h) for (x, y, w, h) in faces]


           # Handle ongoing effect with pause and zoom
           if effect_frames_remaining > 0:
               # Use the same frame to create a pause effect
               paused_frame = current_effect['paused_frame']
               zoomed_frame = current_effect['zoomed_frame']
               bubble = current_effect['bubble']
               position = current_effect['position']


               # Overlay the bubble onto the zoomed frame
               frame_with_bubble = blend_bubble(zoomed_frame.copy(), bubble, position)


               # Write the frame to the output video
               out.write(frame_with_bubble)


               # Optional: Display the frame during processing
               cv2.imshow("Comic Style", frame_with_bubble)


               effect_frames_remaining -= 1
           else:
               # Detect motion and initiate effect if appropriate
               if prev_frame is not None and frame_count - last_effect_frame > effect_cooldown:
                   motion_detected, motion_info = detect_motion(prev_frame, frame)


                   if motion_detected and frame_count - last_effect_frame > min_frames_between_effects:
                       center_x, center_y, w, h = motion_info
                       safe_position = find_safe_position((center_x, center_y, w, h), face_regions, 300, frame.shape)


                       if safe_position:
                           action_text = random.choice(action_texts)
                           bubble = create_action_bubble(action_text)


                           # Create zoomed-in frame
                           zoom_factor = 2.0  # Adjust zoom level as needed
                           x1 = max(0, int(center_x - w * zoom_factor / 2))
                           y1 = max(0, int(center_y - h * zoom_factor / 2))
                           x2 = min(width, int(center_x + w * zoom_factor / 2))
                           y2 = min(height, int(center_y + h * zoom_factor / 2))


                           zoomed_region = comic_frame[y1:y2, x1:x2]
                           zoomed_frame = cv2.resize(zoomed_region, (width, height), interpolation=cv2.INTER_LINEAR)


                           # Store effect data
                           current_effect = {
                               'paused_frame': comic_frame.copy(),
                               'zoomed_frame': zoomed_frame,
                               'bubble': bubble,
                               'position': safe_position
                           }
                           effect_frames_remaining = effect_duration
                           last_effect_frame = frame_count
                   else:
                       # Write the frame to the output video
                       out.write(comic_frame)


                       # Optional: Display the frame during processing
                       cv2.imshow("Comic Style", comic_frame)
               else:
                   # Write the frame to the output video
                   out.write(comic_frame)


                   # Optional: Display the frame during processing
                   cv2.imshow("Comic Style", comic_frame)


           frame_count += 1
           if frame_count % 5 == 0:
               progress = (frame_count / total_frames) * 100
               print(f"\rProgress: {progress:.1f}%", end="")


           prev_frame = frame.copy()
           prev_comic_frame = comic_frame.copy()


           # Wait for a short period (adjust if needed)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break


       print("\nProcessing complete!")
       return True


   except Exception as e:
       print(f"Error processing video: {str(e)}")
       return False


   finally:
       if 'video' in locals():
           video.release()
       if 'out' in locals():
           out.release()
       cv2.destroyAllWindows()


# if __name__ == "__main__":
#    input_video = "/Users/aneesh/Documents/Hack112/pie.mp4"
#    create_dynamic_comic_video(input_video)

def gradientColor(bgr, target):
    diff = [(target[0]-bgr[0])//2, (target[1]-bgr[1])//2, (target[2]-bgr[2])//2]
    bgr[0], bgr[1], bgr[2] = bgr[0]+diff[0], bgr[1]+diff[1], bgr[2]+diff[2]