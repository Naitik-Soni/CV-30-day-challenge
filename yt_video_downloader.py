from yt_dlp import YoutubeDL

URL = "https://www.youtube.com/watch?v=4aWufTZDLMU"
SAVE_PATH = r"C:\Users\baps\Documents\Naitik Soni\ComputerVision\CV-30-day-challenge\Practical tasks day (1-15)\Task-2"   # <-- your folder
RESOLUTION = "1080"                  # example: 720, 1080, 1440, 2160
FPS = "60"                           # example: 30 or 60

ydl_opts = {
    "quiet": True,                   # no unnecessary printing
    "no_warnings": True,
    "outtmpl": f"{SAVE_PATH}/%(title)s.%(ext)s",
    "format": f"bestvideo[height={RESOLUTION}][fps={FPS}]+bestaudio/best"
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])
