import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(BASE_DIR, "camera_mapping.json")

with open(MAPPING_FILE, "r") as f:
    camera_map = json.load(f)


def get_camera_details(filename):
    filename = filename.lower().strip()

    if filename in camera_map:
        return camera_map[filename]

    return {
        "camera_id": "Unknown",
        "room_no": "Unknown Location"
    }