# re4r-speedrun-review-helper
A tool used to help moderators verify speedrun in RE4R

What it does:
Flags timestamps for inventory open in the typewriter and FPS changes in RE4R.

How to use:
Download the releases and extract the files. 
Open the RuleCheck.exe.
Paste the link from the video you wish to analyze the run or if you have it downloaded already, browse the files. 
If it's a link from bilibili, it might ask you to login. Choose the browser you already have logged in on the "Browser cookies" tab.
Click start and wait.

Limitations:
It works best with 1080p videos, not sure how it would handle 4k and 2k. Timestamp accuracy might not be 100%.

How to build it yourself:
Have Python 3.11 installed
Open terminal and use CD command to head to the folder where rulechecker.py is located.
Execute these commands:
pip install opencv-python-headless numpy rapidocr-onnxruntime yt-dlp pyinstaller
pyinstaller --onefile --windowed --name RuleCheck --collect-all rapidocr_onnxruntime --collect-all yt_dlp rulecheck.py


It might get flagged as a virus since it is an unknown .exe!

It uses OpenCV, RapidOCR and yt-dlp. Code was generated 100% with AI.
