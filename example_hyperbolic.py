import base64
import requests
import os
from io import BytesIO
from PIL import Image


def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string


img = Image.open("./example.jpg")
base64_img = encode_image(img)

api = "https://api.hyperbolic.xyz/v1/chat/completions"
api_key = os.getenv("HYPERBOLIC_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}


payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
            },
        ],
    }],
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "max_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.001,
}

response = requests.post(api, headers=headers, json=payload)
print(response.json())
