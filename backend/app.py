#install dependicies
from flask import Flask, jsonify, request, send_file, after_this_request
from flask_cors import CORS
import os
import threading
import cv2 #OpenCV for video reading 
import time
import sys
import easyocr #optical character recog (reading text from images)
import chromeadb #vector based text storage and search
import multiprocessing

#create flask app
app = Flask(__name__)
CORS = (app)

#directory vars
video_dir = "videos"
screenshot_dir = "screenshots"
db_path = "db_data"
confidence_threshold = 0.8 #only save detections with > 80% confidence

#OCR and database setup
print("intializing main ocr reader...")
try: 
    main_reader = easyocr.Reader(['en'], gpu=False) #use CPU for stability in eng lang
    print("Main OCR reader intalized")
except Exception as e:
    print("Did not initialize try again")
    sys.exit()

#ChromDB client setup
print("intializing chromdb...")
try: 
    client = chromadb.PersistentClient(path=db_path) #save and load database from local machine
    print("chromdb intialized")
except Exception as e:
    print(f"Did not intialize try again")
    sys.exit()

#directory and file setup
for directory in [video_ocr, screenshot_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

#functional functions!
def save_text_data(text, metadata):
    #save single text entry to chromadb
    try:
        collection = client.get_collection(name = "video_text_search")
        #extra metadata other than text: camera id, source video, timestamp
        unique_id = f"{metadata['camera_id']}_{metadata['timestamp']}_{metadata['timestamp']}"
        collection.add(documents=[text], metadatas=[metadata], ids=[unique_id]) #add metadatas to my queries later
        return True 
    except Exception as e:
        #skip duplicate ids
        if "ID already exists" not in str(e):
            print(f"error saving data to chromedb: {e}") #otherwise some error we don't know
        return False 
    
#search queries in chromadb
def search_text_data(query):
    #search for text using chromdb with a filter
    try:
        collection = client.get_collection(name = "video_text_search")
        #use where_document = {"$contains": query} for substring search
        results = collection.query(
            query_text = [query], 
            where_document = {"$contains: query"}, 
            num_results = 50
        )
        #return top 50 matches with timestamp, source video, camera id, screenshot filename (for frontend)
        if not results or not results['documents']:
            return []
        
        #formatting for frontend use
        #chromdb.query() returns dictionary 
        """
        'documents': [['MS123', ...]], 'metadatas': [[{'timestamp: 12.5, 'camera_id': cam_1, ...}]]
        """
        formatted_results = []
        for i, info in enumerate(results[documents][0]):#for first query batch
            meta = results['metadats'][0][i]
            formatted_result = {
                'text': info,
                'timestamp': meta.get('timestamp'),
                'camera_id':meta.get('camera_id'),
                'source_video' : meta.get('source_video'),
                'screenshot_filename': os.path.basename(meta.get('screenshot_path', ''))
            }
            formatted_results.append(formatted_result)
        return formatted_results
    
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def capture_screenshot_with_highlight(frame,bbox,text,timestamp, camera_id):
    #don't want to draw on original
    try:
        highlighted_frame = frame.copy()
        #bounding box top left and bottom right corners
        (tl, tr, br, bl) = bbox;
        tl = (int(tl[0], int(tl[1])))
        br = (int(br[0], int(br[1])))
        #draw green box
        cv2.rectangle(highlighted_frame, tl, br, (0, 255, 0), 3)

        #safe filename formatting
        safe_text = "".join(c for c in text if c.isalnum())[:20]
        timestamp_str = f"{timestamp:.1f}s".replace('.', '_')
        filename = f"{safe_text}_{camera_id}_{timestamp_str}.png"
        filepath = os.path.join(screenshot_dir, filename)

        cv2.imwrite(filepath, highlighted_frame)
        print(f" Highlighted screenshot saved: {filename}")
        return filepath

    except Exception as e:
        print(f"Error saving highlighted screenshot: {e}")
        return None

#---Multiprocessing functions---

#Global variable for each worker 
#prep each parallel worker process for OCR tasks during multiprocessing

worker_reader = None #hold an EasyOCR.Reader

def init_worker():
    """Intializer for each worker process in the pool"""
    global worker_reader
    print(f"Initializing EasyOCR reader in worker process {os.getpid()}...")
    # Each process gets its own reader instance to avoid conflicts
    worker_reader = easyocr.Reader(['en'], gpu=False)
    print(f"Worker {os.getpid()} initialized.")


#most important function 
#process a single video file, frame by frame, uses EasyOCR to detect text, and saves result(screenshots + metadata)
def process_video_task(video_path):
    global worker_reader
    if worker_reader is None:
        print("Error: Worker not intialized.")
        return 

    video_file = os.path.basename(video_path) #process one file passed as video path
    camera_id = os.path.splitext(video_file)[0] #use filename as camera_id (remove .mp4)
    interval_seconds = 1 #process 1 frame/sec, only run OCR every 1 second of video time

    try:
        #open video
        cap = cv2.VideoCapture(video_path) #open video using OpenCV
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        #cap = capture refere to an instance of OpenCV's VideoCapture object
        #calculate frame interval in 30 FPS video (default)
        #only analyze 1 frame so skip 29 frames (to save compute)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        frames_to_skip = int(fps * interval_seconds)
    
        #loop thru video frames
        frame_count = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, frame = cap.read()
            if not success:
                break
            #run ocr on frame
            ocr_results = worker_reader.readtext(frame)
            curr_timestamp = frame_count / fps #what time it showed up

            for (bbox, text, confidence) in ocr_results:
                if confidence >= confidence_threshold:
                    #save highlighted screenshot + metadata
                    screenshot_path = os.path.basename(screenshot_dir)
                    metadata = {'timestamp': curr_timestamp,
                                'camera_id': camera_id,
                                'source_video' : video_file,
                                'confidence' : confidence,
                                'screenshot_path' : screenshot_path or ''}
                    save_text_data(text, metadata)
            #move to next frame
            frame_count += frames_to_skip
        print(f"Completed proceswsing {video_file}")
    except Exception as e:
        print(f"Error in processing video for {video_file}: {e}")
    #cleanup
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def process_all_video_multiprocess():
    """Finds all videos and processes them using a multiprocessing pool."""
    print("\n--- Starting Automatic Video Processing using Multiprocessing ---")

    try:
        #cleanup old ocr results from chromaDB
        client.delete_collection(name = "video_text_search")
        client.get_or_create_collection(name="video_text_search")
        print("Collection cleared and recreated.")
    except Exception as e:
        print(f"Could not clear collection (it may not have existed): {e}")
        # Ensure the collection exists for the workers
        client.get_or_create_collection(name="video_text_search")
    
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    if not video_files:
        print("No video files found in the 'videos' directory.")
        return
    
    print(f"Found {len(video_files)} videos to process: {[os.path.basename(f) for f in video_files]}")
    #use multiprocessing pool to process videos in parallel, limit to 4 videos at a time
    num_processes = min(multiprocessing.cpu_count(), 4)
    with multiprocessing.Pool(processes = num_processes, intializer=init_worker) as pool:
        #create a pool of worker process each one runs init worker settingf up easy ocr instance
        pool.map(process_video_task, video_files)
    print("all video processing taasks dispatched")


    #--Flask API routes-- (post to send and get to fetch)
    @app.route('/api/process', methods = ['POST'])
    def process_video():
        #enpoint to processing for single video
        data = request.json
        video_file = data.get('video_file')
        if not video_file:
            return jsonify({'error': 'No video_file provided'}), 400
        video_path = os.path.join(video_dir, video_file)
        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 404
        try:
            #cannot call process_video_task(video_path) directly, API would freeze until finished
        #create bg thread that will run the process_video_task in background without blocking API resposne
            thread = threading.Thread(target=process_video_task, args= (video_path,))
            thread.daemon = True #(thread auto shut down when main program exits)
            thread.start()

            return jsonify({
                "status": "processing", 
                "video": video_file,
                "message": "video processing started in background"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    

    @app.route('/api/search', method=['GET'])
    # Common status codes: 400 = bad request, 500 = Internal server error, 404 = Not Found
    def search_text():
        #endpoint for searching text in videos
        query = request.args.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        try: 
            results = search_text_data(query)
            return jsonify({"results": results})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @app.route('/api/screenshot/<filename>', methods=['GET'])
    def get_screenshot(filename):
        #endpoint for screenshot images

        screenshot_dir = "screenshots"

        if '..' in filename or '/' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join(screenshot_dir, filename)

        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            #return 404, for demo find similar filename
            for file in os.listdir(screenshot_dir) if os.path.exits(screenshot_dir) else[]:
                #return the first image file as a fallback
                if file.endswith(('.png', 'jpg', 'jpeg')):
                    return send_file(os.path.join(screenshot_dir, file), mimetype='image/png')
            return jsonify({"error": "Screenshot not found"}), 404
        

    @app.route('/api/videos', methods=['GET'])
    def list_videos():
        #endpoint to list available videos
        try:
            video_files = [f for f in os.listdir(video_dir)if f.endswith(('.mp4', '.mov', '.avi'))]
            videos = [{"id": f, "name": f, "camera_id": os.path.splitext(f)[0]} for f in video_files]
            return jsonify({"videos":videos})
        except FileNotFoundError:
            return jsonify({"videos": []})

    @app.route('/api/video_segment/<filename>', methods=['GET'])
    def get_video_segment():
        #endpoint to get video segment based on timestamp (returns only the video segment with detected text)
        #get input
        video_path_arg = request.args.get('video_path', '')
        start_time = float(request.args.get('start_time',0))
        end_time = float(request.args.get('end_time',0))
        #validate input
        if not video_path_arg:
            return jsonify({"Error: no path provided"}, 400)
        source_video_path = os.path.join(video_dir, video_path_arg)
        if not os.path.exists(source_video_path):
            return jsonify({"error": f"Video file not found: {video_path_arg}"}, 404)
        #process video using cv2
        try:
            cap = cv2.VideoCapture(source_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #create mp4 file using MPEG-4 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            start_frame = int(start_time*fps)
            end_frame = int(end_time*fps)

            #create temp file to store video segment
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video_path = temp_video.name
            writer= cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height)) #writes the individual video frames into temp file
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame = start_frame
            while current_frame <= end_frame:
                success, frame = cap.read()
                if not success:
                    break
                writer.write(frame)
                current_frame +=1
            cap.release()
            writer.release()
        #clean up (delete temp video file from server)
            @after_this_request
            def cleanup(response):
                try:
                    os.remove(temp_video_path)
                except Exception as e:
                    app.logger.error("Error removing temp file: %s", e)
                return response
            
            return send_file(temp_video_path, mimetype='video/mp4')
        except Exception as e:
            return jsonify({"Error: failed to create video segemnt"}, 500)


    #index() function returns a string that will be displayed in the web browser when someone visits site's root URL
    @app.route('/')
    def index():
        return "MSI Video Tracker API is running! Videos are being processed automatically."

    if __name__ == '__main__':
        #needed on windows
        multiprocessing.freeze_support()
        # Start video processing at startup
        process_all_video_multiprocess()
        # Start the Flask app
        app.run(debug=True, port=3000, use_reloader=False)

    

        


    




    



    






