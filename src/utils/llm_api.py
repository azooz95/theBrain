from utils.image_handler import ImageHandler
from google import genai
from google.genai import types
from typing import Optional

from utils.image_handler import ImageHandler
import PIL.Image

from dotenv import load_dotenv
import os 

load_dotenv()

gemni_api_key = os.getenv("GOOGLE_API_KEY")

def llm_api(contents:Optional[str] = None, wanted_information: Optional[str] = None):
    client = genai.Client(api_key=gemni_api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"Extract {wanted_information} from the following text: {contents}. Return the output in valid JSON format. If any information is missing, use null"])

    return response.text

