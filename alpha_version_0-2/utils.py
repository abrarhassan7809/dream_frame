import os
from datetime import datetime

def save_png(image):
    os.makedirs("outputs", exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    path = os.path.join("outputs", filename)
    image.save(path, "PNG")

    return path