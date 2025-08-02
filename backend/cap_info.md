| Method or Property                              | What it does                             |
| ----------------------------------------------- | ---------------------------------------- |
| `cap.read()`                                    | Reads the next frame from the video      |
| `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)` | Jumps to a specific frame                |
| `cap.get(cv2.CAP_PROP_FPS)`                     | Gets the video’s FPS (frames per second) |
| `cap.get(cv2.CAP_PROP_FRAME_COUNT)`             | Total number of frames in the video      |
| `cap.release()`                                 | Closes the file when you’re done         |
