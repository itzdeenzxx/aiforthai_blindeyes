from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech
from aift.multimodal import vqa
from aift import setting
import requests
import base64
import io
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys
google_api_key = 'AIzaSyBOzx-ERFptAJDmL6ljWCQtLhOKtY457uQ'
aiforthai_api_key = 'kOOudAlAEDw4J2CbSeKZSXRVkpB37Wc3'

# Initialize AI for Thai
setting.set_api_key(aiforthai_api_key)

def speech_to_text_google_api_key(audio_file, api_key):
    try:
        client = speech.SpeechClient(client_options={"api_key": api_key})
        audio_content = audio_file.read()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="th-TH"
        )
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            raise HTTPException(status_code=400, detail="No speech detected in the audio")
        return response.results[0].alternatives[0].transcript
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech to text error: {str(e)}")

def translate_text_google_api_key(text, target_language='en'):
    try:
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {
            'q': text,
            'target': target_language,
            'source': 'th',
            'key': google_api_key
        }
        response = requests.post(url, params=params)
        if not response.ok:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Translation API error: {response.text}"
            )
        result = response.json()
        return result['data']['translations'][0]['translatedText']
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Translation request error: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Invalid translation response format: {str(e)}")

def process_image_with_aiforthai(image_bytes, question):
    try:
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        try:
            # Use AI for Thai VQA
            result = vqa.generate(temp_file_path, question)
            return result.get('content', 'No response from AI')
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI for Thai error: {str(e)}")

@app.post('/process')
async def process_data(audio: UploadFile = File(...), image: UploadFile = File(...)):
    try:
        # Read image content
        image_content = await image.read()
        
        # Process audio
        audio_content = await audio.read()
        audio_file = io.BytesIO(audio_content)
        
        # Get transcript
        transcript = speech_to_text_google_api_key(audio_file, google_api_key)
        print(f"Transcript: {transcript}")
        
        # Process with AI for Thai
        ai_response = process_image_with_aiforthai(image_content, transcript)
        print(f"AI response: {ai_response}")
        
        # Translate the AI response to English (optional)
        translated_response = translate_text_google_api_key(ai_response)
        print(f"Translated response: {translated_response}")
        
        return JSONResponse(content={
            'speech_to_text': transcript,
            'image_description': ai_response,
            'translated_text': translated_response
        })
    except HTTPException as e:
        return JSONResponse(
            content={'error': str(e.detail)},
            status_code=e.status_code
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            content={'error': f'Unexpected error: {str(e)}'},
            status_code=500
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")