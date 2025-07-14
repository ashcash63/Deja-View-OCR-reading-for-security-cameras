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





