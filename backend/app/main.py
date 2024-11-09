from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from pydub.utils import which
import requests
import json
import io
import tempfile
import os
from aift import setting
import logging
import uuid

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

AudioSegment.ffmpeg = which("ffmpeg")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys
aiforthai_api_key = 'kOOudAlAEDw4J2CbSeKZSXRVkpB37Wc3'
botnoi_token = 'bkRQaFB1clAxMGhQd2poWWZxZURETUhuNEtOMjU2MTg5NA=='

setting.set_api_key(aiforthai_api_key)

def process_audio_for_patii(audio_content):
    try:
        # Convert audio to WAV format with required specifications
        audio = AudioSegment.from_file(io.BytesIO(audio_content))
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio.export(temp_audio.name, format='wav')
            
            # Call Patii STT API
            url = "https://api.aiforthai.in.th/partii-webapi"
            files = {'wavfile': ('audio.wav', open(temp_audio.name, 'rb'), 'audio/wav')}
            headers = {
                'Apikey': aiforthai_api_key,
                'Cache-Control': "no-cache",
                'Connection': "keep-alive",
            }
            params = {"outputlevel": "--uttlevel", "outputformat": "--txt"}
            
            response = requests.post(url, headers=headers, files=files, data=params)
            os.unlink(temp_audio.name)
            
            if response.ok:
                return response.json()['message']
            else:
                raise HTTPException(status_code=response.status_code, detail="Patii STT API error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

def process_image_with_capgen(image_content):
    try:
        # Create temporary file for the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_content)
            temp_file_path = temp_file.name

            url = 'https://api.aiforthai.in.th/capgen'
            headers = {'Apikey': aiforthai_api_key}
            files = [('file', ('image.jpg', open(temp_file_path, 'rb'), 'image/jpeg'))]
            
            response = requests.post(url, headers=headers, files=files)
            os.unlink(temp_file_path)
            
            if response.ok:
                return response.json()['caption']
            else:
                raise HTTPException(status_code=response.status_code, detail="Capgen API error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def process_with_patumma(question, description, sessionid):
    try:
        
        combined_input = f"ให้สมมติว่านี่คือสิ่งที่คุณมองเห็น : ({description}) และคุณคือเพื่อนสนิทของฉันมากๆ และคุณพร้อมจะตอบในสิ่งที่คุณมองเห็น \nโดยที่ฉันจะถามว่า: {question}"
        
        # Debugging check to ensure combined_input is not empty
        if not combined_input.strip():
            logger.error("Combined input is empty or whitespace. Cannot send request to Patumma.")
            raise HTTPException(status_code=400, detail="Combined input is empty")

        response = chat(instruction=combined_input, sessionid=sessionid, temperature=0.4)

        logger.debug(f"Patumma API response: {response}")
        return response.get('response', 'No response found from Patumma.')

    except Exception as e:
        logger.error(f"Patumma processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Patumma processing error: {str(e)}")

def chat(instruction: str, 
         sessionid: str,
         context: str = "", 
         temperature: float = 0.4,
         return_json: bool = True):

    headers = {'accept': 'application/json', 'Apikey': aiforthai_api_key, 'X-lib': 'ai4thai-lib'}
    
    # Patumma API URL
    url = 'https://api.aiforthai.in.th/pathumma-chat'
    
    payload = {
        'context': context,
        'prompt': instruction,
        'sessionid': sessionid,
        'temperature': temperature,
    }

    try:
        res = requests.post(url, headers=headers, data=payload)
        res.raise_for_status()

        if not return_json:
            return res.json()['response']

        return res.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error with Patumma API request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Patumma API request error: {str(e)}")

def text_to_speech(text: str):
    url = "https://api-voice.botnoi.ai/openapi/v1/generate_audio"
    payload = {
        "text": text,
        "speaker": "26",  # You can adjust the speaker ID as needed
        "volume": 1,
        "speed": 1,
        "type_media": "m4a",
        "save_file": "true",
        "language": "th"
    }
    headers = {
        'Botnoi-Token': botnoi_token,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("audio_url", None)
    else:
        raise HTTPException(status_code=response.status_code, detail="Botnoi text-to-speech API error")

@app.post('/process')
async def process_data(audio: UploadFile = File(...), image: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        logger.debug(f"Generated session ID: {session_id}")
        
        logger.debug("Received audio and image files.")
        
        image_content = await image.read()
        audio_content = await audio.read()
        
        logger.debug(f"Audio size: {len(audio_content)} bytes, Image size: {len(image_content)} bytes")

        transcript = process_audio_for_patii(audio_content)
        logger.debug(f"Transcript: {transcript}")
        
        image_description = process_image_with_capgen(image_content)
        logger.debug(f"Image description: {image_description}")
        
        llm_response = process_with_patumma(transcript, image_description, session_id)
        logger.debug(f"LLM response: {llm_response}")
        
        audio_response = text_to_speech(llm_response)
        logger.debug(f"Generated audio response.")

        return JSONResponse(content={
            'speech_to_text': transcript,
            'image_description': image_description,
            'llm_response': llm_response,
            'audio_response': audio_response,
            'session_id': session_id  # Include session ID in response
        })
        
    except HTTPException as e:
        logger.error(f"HTTP error: {str(e.detail)}")
        return JSONResponse(
            content={'error': str(e.detail)},
            status_code=e.status_code
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return JSONResponse(
            content={'error': f'Unexpected error: {str(e)}'},
            status_code=500
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")