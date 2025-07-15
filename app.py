#install dependicies
from flask import Flash, jsonify, request, send_file, after_this_request
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
#video_dir = "videos"
#screenshot_dir = "screenshots"
#db_path = "db_data"
#confidence_threshold = 0.8 #only save detections with > 80% confidence

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
    #search fro text using chromdb with a filter
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
                'camera_id' = meta.get('camera_id'),
                'source_video' = meta.get('source_video'),
                'screenshot_filename': os.path.basename(meta.get('screenshot_path', ''))
            }
            formatted_results.append(formatted_result)
        return formatted_results
    
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

    def capture_screenshot_with_highlight(frame,bbox,text,timestamp, camera_id):
        #don't want to draw on original
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


    






