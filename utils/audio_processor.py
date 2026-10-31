import yt_dlp
from pydub import AudioSegment
import os
import shutil
from pathlib import Path

DOWNLOAD_DIR = 'downloads'
#makes directory if not exists
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

# finds the location of ffmpeg and ffprobe executables, 
# either from the system PATH or from the WinGet installation directory. 
# Raises a FileNotFoundError if not found.
#processing audio using ffmpeg and ffprobe, which are required for audio processing tasks.

def _find_ffmpeg_location() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        return str(Path(ffmpeg_path).parent)

    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for package_dir in winget_root.glob("Gyan.FFmpeg.Shared_*"):
        for bin_dir in package_dir.rglob("bin"):
            if (bin_dir / "ffmpeg.exe").exists() and (bin_dir / "ffprobe.exe").exists():
                return str(bin_dir)

    raise FileNotFoundError(
        "FFmpeg and ffprobe were not found. Install FFmpeg or set FFMPEG_LOCATION."
    )

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s") #file name and extension
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": os.environ.get("FFMPEG_LOCATION") or _find_ffmpeg_location(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,#to check download progress
    }
    # opening file with with closes it when not in use
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    #sets dual to mono audio nd convert to 16khz sample rate for better processing in Whisper model. Whisper model performs better with mono audio at 16kHz sample rate.
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

if __name__ == "__main__":
    data = download_youtube_audio("https://www.youtube.com/watch?v=mtiOK2QG9Q0")
    
#data_final = convert_to_wav(data)    

#chunking the audio
#returns path of chunks given to 
#chunk_minutes is the step = 10 mins
def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 #convert to milli secs min to milli sec

    chunks = []
# start from 0 to length of audio in steps of milli sec
    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        #slicing audio
        chunk = audio[start : start + chunk_ms] #chunk_ms is the step
        chunk_path = f"{wav_path}_chunk_{i}.wav" #i is index starts from 0
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks
#print(chunk_audio(data_final))

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks