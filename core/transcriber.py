import os
import requests
import time
from pydub import AudioSegment
from utils.audio_processor import _find_ffmpeg_location

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
SARVAM_TIMEOUT = (
    int(os.getenv("SARVAM_CONNECT_TIMEOUT", "60")),
    int(os.getenv("SARVAM_READ_TIMEOUT", "300")),
)
SARVAM_MAX_RETRIES = int(os.getenv("SARVAM_MAX_RETRIES", "3"))

_model = None

def _ensure_ffmpeg_on_path() -> None:
    ffmpeg_location = os.environ.get("FFMPEG_LOCATION") or _find_ffmpeg_location()
    current_path = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_location not in current_path:
        os.environ["PATH"] = ffmpeg_location + os.pathsep + os.environ.get("PATH", "")

#loading whisper model only once and reusing it for all chunks
def load_model():

    global _model  

    if _model is None: 
        import whisper

        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str) -> str:
    _ensure_ffmpeg_on_path()
    model = load_model()  

    result = model.transcribe(chunk_path, task="transcribe")  
    return result["text"]  


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    for attempt in range(SARVAM_MAX_RETRIES):
        try:
            with open(piece_path, "rb") as f:
                files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
                data = {"model": SARVAM_MODEL, "with_diarization": "false"}
                response = requests.post(
                    SARVAM_STT_TRANSLATE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=SARVAM_TIMEOUT,
                )
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == SARVAM_MAX_RETRIES - 1:
                raise
            delay = 2 ** attempt
            print(f"  Sarvam request failed; retrying in {delay}s ({attempt + 1}/{SARVAM_MAX_RETRIES - 1})...")
            time.sleep(delay)

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


# def transcribe_chunk_sarvam(chunk_path: str) -> str:
#     """
#     Sarvam sync API only accepts ≤30s audio. We split this chunk into
#     25-second pieces, send each separately, and join the transcripts.
#     """
#     if not SARVAM_API_KEY:
#         raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    # audio = AudioSegment.from_wav(chunk_path)
    # piece_ms = SARVAM_PIECE_SECONDS * 1000

    # full_text = ""
    # total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    # for i, start in enumerate(range(0, len(audio), piece_ms)):
    #     piece = audio[start: start + piece_ms]
    #     piece_path = f"{chunk_path}_sv_{i}.wav"
    #     piece.export(piece_path, format="wav")

    #     try:
    #         print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
    #         full_text += _send_to_sarvam(piece_path) + " "
    #     except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as error:
    #         print(f"\n⚠️ Sarvam is unreachable ({error}). Falling back to local Whisper.")
    #         return transcribe_chunk_whisper(chunk_path)
    #     finally:
    #         if os.path.exists(piece_path):
    #             os.remove(piece_path)

    #return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "tanglish":
        return 0
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = "" 

    engine = "Sarvam AI" if language.lower() == "tanglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
#get the text from the chunk using the appropriate transcription method
        text = transcribe_chunk(chunk, language=language)  

        full_transcript += text + " "  

    print("Transcription complete.")

    return full_transcript.strip()  