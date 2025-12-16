import os, re

def natural_key(text):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', text)]

def get_files_list(folder_path, ext=".png"):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(ext)
    ]

    files_sorted = sorted(files, key=lambda x: natural_key(os.path.basename(x)))

    return files_sorted[::-1]