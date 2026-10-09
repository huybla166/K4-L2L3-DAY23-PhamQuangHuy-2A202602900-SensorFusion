"""Import the bonus CVAT export into a local CVAT and capture the evidence screenshots.

Prerequisites (outside the lab environment):

    * CVAT running locally (``docker compose up -d`` in a CVAT checkout, http://localhost:8080)
      with an account whose name/password are in ``CVAT_USER`` / ``CVAT_PASSWORD``;
    * ``pip install requests playwright pillow`` and ``playwright install chromium``;
    * ``python student/bonus/bonus_analysis.py cvat`` already run (BEV frames in
      ``data/cache/bev_cvat/`` and ``student/bonus/cvat/annotations_cvat_video_1.1.xml``).

The script creates one task from the 199 BEV frames, imports the XML in the
"CVAT 1.1" format through the REST API, checks how many tracks CVAT stored, and
saves screenshots of the annotation view at the ghost / ID-change frames listed in
``student/bonus/cvat/cvat_events.json``.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

HOST = os.environ.get("CVAT_HOST", "http://localhost:8080")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "student" / "bonus" / "cvat"
FRAMES = ROOT / "data" / "cache" / "bev_cvat"
XML = OUT / "annotations_cvat_video_1.1.xml"

LABELS = [
    {"name": "fusion_track", "color": "#ff9900", "type": "rectangle", "attributes": [
        {"name": "track_id", "mutable": False, "input_type": "text",
         "default_value": "-1", "values": ["-1"]},
        {"name": "state", "mutable": True, "input_type": "select",
         "default_value": "initialized", "values": ["initialized", "tentative", "confirmed"]},
        {"name": "ghost", "mutable": True, "input_type": "checkbox",
         "default_value": "false", "values": ["false"]},
    ]},
    {"name": "gt_vehicle", "color": "#33dd33", "type": "rectangle", "attributes": [
        {"name": "gt_id", "mutable": False, "input_type": "text", "default_value": "-", "values": ["-"]},
    ]},
]

# (output name, panels side by side as (frame, focused track_id, focused gt_id), caption per panel).
# A panel without focus shows the text of every object; a focused panel shows only the
# hovered fusion_track on the canvas (labels of overlapping boxes would cover each other)
# and expands the attributes of that track and of its GT vehicle in the sidebar.
VJ3F, EFR8 = "VJ3F-NiHmRQh0umylCVJmA", "8EFRSwEXBf9P-SzIDRzx0A"
SHOTS = [
    ("cvat_ghost_track4_frame30.png", [(30, None, None)],
     ["frame 30: fusion_track #4 (tentative, ghost=true) with no gt_vehicle box"]),
    ("cvat_id_change_5_to_7.png", [(58, 5, VJ3F), (60, 7, VJ3F)],
     ["frame 58: GT VJ3F... tracked as track_id 5 (confirmed)",
      "frame 60: same GT re-initialised as track_id 7"]),
    ("cvat_id_change_8_9_10.png", [(85, 8, EFR8), (88, 9, EFR8), (92, 10, EFR8)],
     ["frame 85: GT 8EFR... -> track_id 8", "frame 88: same GT -> track_id 9",
      "frame 92: same GT -> track_id 10"]),
]


def client_ids() -> dict[str, int]:
    """CVAT numbers objects in the order of the imported XML tracks, starting at 1."""
    import xml.etree.ElementTree as ET

    ids = {}
    for track in ET.parse(XML).getroot().findall("track"):
        attrs = {a.get("name"): a.text for a in track.find("box").findall("attribute")}
        key = attrs.get("gt_id") or f"track_{attrs['track_id']}"
        ids[key] = int(track.get("id")) + 1
    return ids


def session() -> requests.Session:
    s = requests.Session()
    s.auth = (os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"])
    return s


def wait_request(s: requests.Session, rq_id: str) -> None:
    for _ in range(600):
        r = s.get(f"{HOST}/api/requests/{rq_id}")
        r.raise_for_status()
        status = r.json()["status"]
        if status == "finished":
            return
        if status == "failed":
            raise RuntimeError(r.json())
        time.sleep(2)
    raise TimeoutError(rq_id)


def create_task(s: requests.Session) -> int:
    r = s.post(f"{HOST}/api/tasks", json={"name": "day23_bev_fused_tracks", "labels": LABELS})
    r.raise_for_status()
    task_id = r.json()["id"]
    files = sorted(FRAMES.glob("*.png"))
    upload = [(f"client_files[{i}]", (p.name, p.read_bytes(), "image/png")) for i, p in enumerate(files)]
    r = s.post(f"{HOST}/api/tasks/{task_id}/data", files=upload,
               data={"image_quality": 95, "sorting_method": "natural"})
    r.raise_for_status()
    wait_request(s, r.json()["rq_id"])
    return task_id


def import_annotations(s: requests.Session, task_id: int) -> dict:
    r = s.post(f"{HOST}/api/tasks/{task_id}/annotations", params={"format": "CVAT 1.1"},
               files={"annotation_file": (XML.name, XML.read_bytes(), "application/xml")})
    r.raise_for_status()
    wait_request(s, r.json()["rq_id"])
    r = s.get(f"{HOST}/api/tasks/{task_id}/annotations")
    r.raise_for_status()
    data = r.json()
    labels = {l["id"]: l["name"] for l in s.get(f"{HOST}/api/labels", params={"task_id": task_id}).json()["results"]}
    counts: dict[str, int] = {}
    for track in data["tracks"]:
        name = labels.get(track["label_id"], str(track["label_id"]))
        counts[name] = counts.get(name, 0) + 1
    return counts


def screenshots(task_id: int, job_id: int) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.goto(f"{HOST}/auth/login")
        page.wait_for_selector("input#credential", timeout=60000)
        page.fill("input#credential", os.environ["CVAT_USER"])
        page.fill("input#password", os.environ["CVAT_PASSWORD"])
        page.keyboard.press("Enter")
        page.wait_for_url(lambda url: "/auth/login" not in url, timeout=60000)
        ids = client_ids()
        for name, panels, captions in SHOTS:
            images = []
            for frame, track_id, gt_id in panels:
                page.goto(f"{HOST}/tasks/{task_id}/jobs/{job_id}?frame={frame}")
                page.wait_for_selector(".cvat-canvas-container", timeout=60000)
                page.wait_for_timeout(2500)
                # Settings → Workspace → always show object details (label + attributes)
                page.keyboard.press("F2")
                page.get_by_role("tab", name="Workspace").click()
                show_all = page.locator(".cvat-workspace-settings-show-text-always input")
                show_all.set_checked(track_id is None)
                page.keyboard.press("Escape")
                page.wait_for_timeout(500)
                frame_input = page.locator(".cvat-player-frame-selector input").first
                if frame_input.input_value() != str(frame):
                    frame_input.fill(str(frame))
                    frame_input.press("Enter")
                page.wait_for_timeout(2500)
                if track_id is not None:
                    items = [page.locator(f"#cvat-objects-sidebar-state-item-{ids[key]}")
                             for key in (f"track_{track_id}", gt_id)]
                    for item in items:  # expand DETAILS: track_id/state/ghost and gt_id
                        header = item.locator(".cvat-objects-sidebar-state-item-collapse .ant-collapse-header")
                        if header.get_attribute("aria-expanded") != "true":
                            header.click()
                            page.wait_for_timeout(300)
                    items[0].scroll_into_view_if_needed()
                    items[0].hover()  # activates the track: its text is drawn on the canvas
                    page.wait_for_timeout(1000)
                images.append(Image.open(io.BytesIO(page.screenshot())))
            width = sum(i.width for i in images)
            sheet = Image.new("RGB", (width, images[0].height + 40), "white")
            draw = ImageDraw.Draw(sheet)
            x = 0
            for image, caption in zip(images, captions):
                sheet.paste(image, (x, 40))
                draw.text((x + 10, 12), caption, fill="black")
                x += image.width
            sheet.save(OUT / name)
            print("saved", OUT / name)
        browser.close()


def main() -> None:
    s = session()
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else create_task(s)
    if len(sys.argv) <= 1:
        print("task", task_id, "imported tracks per label:", import_annotations(s, task_id))
    job_id = s.get(f"{HOST}/api/jobs", params={"task_id": task_id}).json()["results"][0]["id"]
    print(json.dumps({"task_id": task_id, "job_id": job_id, "job_url": f"{HOST}/tasks/{task_id}/jobs/{job_id}"}))
    screenshots(task_id, job_id)


if __name__ == "__main__":
    main()
