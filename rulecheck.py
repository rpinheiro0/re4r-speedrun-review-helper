#!/usr/bin/env python3
"""
RE4R run triage tool (GUI + CLI).
  1) Timestamps for when the inventory opens from the typewriter menu (template matching)
  2) FPS counter OCR (auto-detects corner) -> timestamps where FPS changes
All results are "look here" flags for a human moderator, not verdicts.
Run with no arguments for the window, or:  rulecheck.py <link-or-file> [options]
"""
import argparse, csv, glob, os, queue, re, shutil, sys, tempfile, threading, time
from statistics import median
import cv2
import numpy as np

W, H = 1920, 1080
CORNERS = ["tl", "tr", "bl", "br"]
CORNER_W, CORNER_H = 0.22, 0.12
_ocr = None


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def hms(t):
    t = int(t)
    return f"{t//3600:d}:{t%3600//60:02d}:{t%60:02d}"


def link(url, t):
    if not url:
        return ""
    s = max(0, int(t) - 3)
    if "twitch" in url:
        return f"{url.split('?')[0]}?t={s//3600}h{s%3600//60}m{s%60}s"
    if "bilibili" in url:
        return f"{url}{'&' if '?' in url else '?'}t={s}"
    return f"{url}{'&' if '?' in url else '?'}t={s}s"


def roi_box(corner):
    bw, bh = int(W * CORNER_W), int(H * CORNER_H)
    x = 0 if corner[1] == "l" else W - bw
    y = 0 if corner[0] == "t" else H - bh
    return x, y, bw, bh


def get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def read_fps(frame, corner):
    x, y, bw, bh = roi_box(corner)
    crop = cv2.resize(frame[y:y + bh, x:x + bw], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        result, _ = get_ocr()(crop)
    except Exception:
        return None
    if not result:
        return None
    text = " ".join(r[1] for r in result)
    for m in re.findall(r"\d{2,3}", text):
        n = int(m)
        if 10 <= n <= 500:
            return n
    return None


def detect_corner(cap, n=30):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    score = {c: 0 for c in CORNERS}
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (0.02 + 0.96 * k / n)))
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(f, (W, H))
        for c in CORNERS:
            if read_fps(f, c) is not None:
                score[c] += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return max(score, key=score.get), score


def load_templates(folder):
    tps = []
    for p in sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        tps.append((os.path.basename(p), im))
    return tps


def download(url, cookies, log):
    import yt_dlp
    tmp = tempfile.mkdtemp(prefix="rulecheck_")
    opts = {
        "format": "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]/b[height<=1080]/b",
        "outtmpl": os.path.join(tmp, "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [lambda d: d.get("status") == "finished" and log("Download finished.")],
    }
    if cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    log("Downloading video (this can take a few minutes)...")
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Download failed: {e}\nFor Bilibili/YouTube sign-in problems, set 'Browser cookies' "
                           f"to the browser you're logged into.")
    files = glob.glob(os.path.join(tmp, "video.*"))
    if not files:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("Download produced no file.")
    return files[0], tmp


def analyze(a, log=print):
    """Returns (csv_path, rows). Raises RuntimeError on problems."""
    video_path, tmpdir = a.video, None
    if a.video.lower().startswith(("http://", "https://")):
        video_path, tmpdir = download(a.video, a.cookies, log)
        if not a.url:
            a.url = a.video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open the video.")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, test = cap.read()
        if not ok:
            raise RuntimeError("Video opened but frames can't be decoded (probably AV1). "
                               "Download an H.264 version of the video and use the file option.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        log(f"Video: {total/vfps/60:.1f} min @ {vfps:.2f} fps")

        menu_on, fps_on = not a.no_menu, not a.no_fps
        templates = []
        if menu_on:
            tdir = a.templates if os.path.isabs(a.templates) else os.path.join(app_dir(), a.templates)
            templates = load_templates(tdir)
            if not templates:
                log(f"No templates found in '{tdir}', skipping inventory detection.")
                menu_on = False
        if not menu_on and not fps_on:
            raise RuntimeError("Nothing to check.")

        corner = None
        if fps_on:
            if a.corner and a.corner != "auto":
                corner = a.corner
            else:
                log("Detecting FPS counter corner...")
                corner, sc = detect_corner(cap)
                log(f"  hits per corner: {sc} -> using '{corner}'")
                if sc[corner] < 8:
                    log("  WARNING: weak detection. Pick the corner manually.")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(app_dir(), "reports", stamp)
        os.makedirs(os.path.join(outdir, "thumbs"), exist_ok=True)

        menu_every = max(1, round(vfps * a.step))
        ocr_every = max(1, round(vfps * 1.0))
        inv, last_hit, samples = [], None, []
        best = {n: 0.0 for n, _ in templates}
        i, report_every = 0, max(1, int(vfps * 300))
        while cap.grab():
            do_menu = menu_on and i % menu_every == 0
            do_ocr = fps_on and i % ocr_every == 0
            if do_menu or do_ocr:
                ok, f = cap.retrieve()
                if ok:
                    f = cv2.resize(f, (W, H))
                    t = i / vfps
                    if do_menu:
                        small = cv2.cvtColor(cv2.resize(f, (960, 540)), cv2.COLOR_BGR2GRAY)
                        hit = False
                        for name, tp in templates:
                            s = float(cv2.matchTemplate(small, tp, cv2.TM_CCOEFF_NORMED).max())
                            best[name] = max(best[name], s)
                            if s >= a.thresh:
                                hit = True
                        if hit:
                            if last_hit is None or t - last_hit > a.gap:
                                inv.append([t, t])
                                cv2.imwrite(os.path.join(outdir, "thumbs", f"inventory_{int(t):06d}.jpg"), f)
                            else:
                                inv[-1][1] = t
                            last_hit = t
                    if do_ocr:
                        samples.append((t, read_fps(f, corner)))
            i += 1
            if i % report_every == 0:
                log(f"  ...{i/vfps/60:.0f} of {total/vfps/60:.0f} min processed")
        cap.release()

        rows = []
        for s, e in inv:
            dur = e - s + a.step
            flag = f"LONG ({dur:.0f}s)" if dur > a.long else f"{dur:.0f}s"
            rows.append((s, "INVENTORY OPENED (typewriter)", f"visible ~{flag}"))

        if fps_on:
            vals = [(t, v) for t, v in samples if v is not None]
            w = a.fps_window
            cands = []
            for k in range(w, len(vals) - w + 1):
                before = median(v for _, v in vals[k - w:k])
                after = median(v for _, v in vals[k:k + w])
                if abs(after - before) >= a.fps_delta:
                    cands.append((vals[k][0], before, after))
            clusters = []
            for c in cands:
                if clusters and c[0] - clusters[-1][-1][0] <= 10:
                    clusters[-1].append(c)
                else:
                    clusters.append([c])
            for cl in clusters:
                rows.append((cl[0][0], "FPS CHANGE", f"{cl[0][1]:.0f} -> {cl[-1][2]:.0f}"))
            start = None
            for t, v in samples + [(None, 1)]:
                if v is None and start is None:
                    start = t
                elif v is not None and start is not None:
                    end = samples[-1][0] if t is None else t
                    if end - start >= a.missing:
                        rows.append((start, "FPS COUNTER UNREADABLE", f"{end-start:.0f}s"))
                    start = None

        rows.sort()
        csv_path = os.path.join(outdir, "review_report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["time", "seconds", "type", "detail", "link"])
            for t, typ, det in rows:
                wr.writerow([hms(t), f"{t:.1f}", typ, det, link(a.url, t)])

        log("\n===== REVIEW LIST =====")
        for t, typ, det in rows:
            log(f"{hms(t):>8}  {typ:<30} {det}")
        log(f"\n{len(rows)} flags. Saved to: {csv_path}")
        if menu_on:
            log("Best template scores (tune the threshold with these): "
                + str({k: round(v, 2) for k, v in best.items()}))
        if tmpdir and a.keep:
            kept = os.path.join(outdir, os.path.basename(video_path))
            shutil.copy(video_path, kept)
            log(f"Video kept at: {kept}")
        return csv_path, rows
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def make_parser():
    ap = argparse.ArgumentParser(description="RE4R run triage tool")
    ap.add_argument("video", nargs="?", default="", help="video link or file path")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--thresh", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--long", type=float, default=10)
    ap.add_argument("--corner", default="auto", choices=["auto"] + CORNERS)
    ap.add_argument("--fps-delta", type=float, default=20)
    ap.add_argument("--fps-window", type=int, default=5)
    ap.add_argument("--missing", type=float, default=5)
    ap.add_argument("--url", default="")
    ap.add_argument("--cookies", default="", help="chrome, firefox, edge...")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-menu", action="store_true")
    ap.add_argument("--no-fps", action="store_true")
    return ap


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox

    root = tk.Tk()
    root.title("RE4R Run Review Helper")
    root.geometry("760x560")
    q = queue.Queue()
    defaults = make_parser().parse_args([])

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="x")
    frm.columnconfigure(1, weight=1)

    v_video = tk.StringVar()
    v_cookies = tk.StringVar(value="")
    v_corner = tk.StringVar(value="auto")
    v_thresh = tk.StringVar(value=str(defaults.thresh))
    v_delta = tk.StringVar(value=str(defaults.fps_delta))
    v_menu = tk.BooleanVar(value=True)
    v_fps = tk.BooleanVar(value=True)

    ttk.Label(frm, text="Video link or file:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=v_video).grid(row=0, column=1, sticky="ew", padx=5)
    ttk.Button(frm, text="Browse...", command=lambda: v_video.set(
        filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.webm *.mov *.flv"), ("All", "*.*")])
        or v_video.get())).grid(row=0, column=2)

    ttk.Label(frm, text="Browser cookies:").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Combobox(frm, textvariable=v_cookies, width=12, state="readonly",
                 values=["", "chrome", "firefox", "edge", "brave"]).grid(row=1, column=1, sticky="w", padx=5)
    ttk.Label(frm, text="(only if YouTube/Bilibili asks for login)").grid(row=1, column=1, sticky="e")

    opt = ttk.Frame(frm)
    opt.grid(row=2, column=0, columnspan=3, sticky="w", pady=4)
    ttk.Checkbutton(opt, text="Inventory check", variable=v_menu).pack(side="left", padx=4)
    ttk.Checkbutton(opt, text="FPS check", variable=v_fps).pack(side="left", padx=4)
    ttk.Label(opt, text="Counter corner:").pack(side="left", padx=(12, 2))
    ttk.Combobox(opt, textvariable=v_corner, width=6, state="readonly",
                 values=["auto"] + CORNERS).pack(side="left")
    ttk.Label(opt, text="Match threshold:").pack(side="left", padx=(12, 2))
    ttk.Entry(opt, textvariable=v_thresh, width=5).pack(side="left")
    ttk.Label(opt, text="FPS change size:").pack(side="left", padx=(12, 2))
    ttk.Entry(opt, textvariable=v_delta, width=5).pack(side="left")

    btns = ttk.Frame(frm)
    btns.grid(row=3, column=0, columnspan=3, sticky="w", pady=6)
    start_btn = ttk.Button(btns, text="Start")
    start_btn.pack(side="left")
    ttk.Button(btns, text="Open reports folder", command=lambda: (
        os.makedirs(os.path.join(app_dir(), "reports"), exist_ok=True),
        os.startfile(os.path.join(app_dir(), "reports")) if sys.platform == "win32"
        else os.system(f'xdg-open "{os.path.join(app_dir(), "reports")}"'))).pack(side="left", padx=8)

    box = scrolledtext.ScrolledText(root, state="disabled", font=("Consolas", 9))
    box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def log(msg):
        q.put(str(msg))

    def pump():
        try:
            while True:
                m = q.get_nowait()
                if m == "\0DONE":
                    start_btn.config(state="normal")
                    continue
                box.config(state="normal")
                box.insert("end", m + "\n")
                box.see("end")
                box.config(state="disabled")
        except queue.Empty:
            pass
        root.after(150, pump)

    def work(a):
        try:
            analyze(a, log)
        except Exception as e:
            log(f"\nERROR: {e}")
        finally:
            q.put("\0DONE")

    def start():
        if not v_video.get().strip():
            messagebox.showinfo("Missing video", "Paste a video link or choose a file first.")
            return
        a = make_parser().parse_args([])
        a.video = v_video.get().strip().strip('"')
        a.cookies = v_cookies.get()
        a.corner = v_corner.get()
        a.no_menu = not v_menu.get()
        a.no_fps = not v_fps.get()
        try:
            a.thresh = float(v_thresh.get())
            a.fps_delta = float(v_delta.get())
        except ValueError:
            messagebox.showerror("Invalid number", "Threshold and FPS change size must be numbers.")
            return
        start_btn.config(state="disabled")
        box.config(state="normal"); box.delete("1.0", "end"); box.config(state="disabled")
        threading.Thread(target=work, args=(a,), daemon=True).start()

    start_btn.config(command=start)
    log("Put your template images in the 'templates' folder next to this program, paste a link, and press Start.")
    pump()
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = make_parser().parse_args()
        try:
            analyze(args)
        except RuntimeError as e:
            sys.exit(str(e))
    else:
        run_gui()