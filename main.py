import os
import json
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

import streamlit as st
import requests

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

with open("secret.json", encoding="utf-8") as f:
    secret = json.load(f)

VISION_ENDPOINT = st.secrets["VISION_ENDPOINT"]
VISION_KEY = st.secrets["VISION_KEY"]
# VISION_ENDPOINT = secret["VISION_ENDPOINT"]
# VISION_KEY = secret["VISION_KEY"]
#os.environ["VISION_ENDPOINT"] = secret["VISION_ENDPOINT"]
#os.environ["VISION_KEY"] = secret["VISION_KEY"]

endpoint = VISION_ENDPOINT
key = VISION_KEY

client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

def get_tags(filepath):
    with open(filepath, "rb") as f:
        image_data = f.read()
        
    result_tags = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.TAGS],
        language="en"
    )
    tag_list = result_tags.tags["values"]
    tags_name = [tag.name for tag in tag_list]
    
    return tags_name

def detect_objects(filepath):
    with open(filepath, "rb") as f:
        image_data = f.read()
    result_objects = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.OBJECTS],
        language="en"
    )
    objects = result_objects.objects["values"]
    return objects

FONT_PATH = "fonts/DejaVuSans.ttf"
FONT_URL = "https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans.ttf"

@st.cache_resource
def load_font(size=16):
    os.makedirs("fonts", exist_ok=True)

    if not os.path.exists(FONT_PATH):
        r = requests.get(FONT_URL)
        r.raise_for_status()
        with open(FONT_PATH, "wb") as f:
            f.write(r.content)
    return ImageFont.truetype(FONT_PATH, size=size)

st.title("Object Detection using Azure Computer Vision")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    os.makedirs("img", exist_ok=True)

    img = Image.open(uploaded_file).convert("RGB")
    img_path = os.path.join("img", uploaded_file.name)
    img.save(img_path)
    
    objects = detect_objects(img_path)
    
    draw = ImageDraw.Draw(img)
    
    # 画像サイズ基準でフォントサイズ決定
    img_w, img_h = img.size
    base = min(img_w, img_h)
    font_size = max(12, min(int(base * 0.025), 32))

    font = load_font(size=font_size)

    # padding もフォント比例
    pad_lr = int(font_size * 0.4)
    pad_tb = int(font_size * 0.25)
    
    for obj in objects:
        box = obj["boundingBox"]
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]
        
        # 物体枠
        draw.rectangle(
            [x, y, x + w, y + h],
            outline="green",
            width=3
        )
        
        tags = ", ".join([tag["name"] for tag in obj["tags"]])
        
        # ラベルは枠の上に配置
        text_x = x
        text_y = max(0, y)
        
        bbox = draw.textbbox(
            (x, y),
            tags,
            font=font,
            anchor="lt"
        )
        
        bg_bbox = (
            bbox[0],
            bbox[1],
            bbox[2] + pad_lr * 2,
            bbox[3] + pad_tb * 2,
        )

        # 背景
        draw.rectangle(bg_bbox, fill="green")

        # 文字
        draw.text(
            (x + pad_lr, y + pad_tb),
            tags,
            fill="white",
            font=font,
            anchor="lt"
        )

    st.image(img, caption="Uploaded Image")
    
    tags_name = get_tags(img_path)

    st.markdown("## Detected Tags:")
    st.markdown(f"> {', '.join(tags_name)}")
    
