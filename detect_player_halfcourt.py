import cv2
import csv
import argparse
import numpy as np
from ultralytics import YOLO


def find_net_line(frame):
    """
    Detect an (approximately) horizontal tennis net line using HoughLinesP.

    Returns:
        (p1, p2): Two points (x, y) defining the chosen line segment.
    Fallback:
        If no suitable line is found, returns a horizontal line at y = 0.55 * H.
    """
    # Convert to grayscale and smooth to reduce noise before edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection (tune thresholds if net is missed)
    edges = cv2.Canny(gray, 60, 160)

    H, W = gray.shape[:2]

    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(0.35 * W),
        maxLineGap=30
    )

    # Fallback if Hough fails
    if lines is None:
        y = int(0.55 * H)
        return (0, y), (W - 1, y)

    best = None
    best_score = -1
    target_y = 0.55 * H  # heuristic: net tends to be around mid-height

    # Score candidates: prefer long, near-horizontal segments close to target_y
    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(int, l)
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5

        # Keep only sufficiently long segments
        if length < 0.35 * W:
            continue

        # Keep only near-horizontal segments
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) > 15:
            continue

        # Prefer segments whose midpoint y is near target_y
        y_mid = 0.5 * (y1 + y2)
        score = length * (1.0 - min(1.0, abs(y_mid - target_y) / H))

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    # Fallback if no candidate survived filtering
    if best is None:
        y = int(0.55 * H)
        return (0, y), (W - 1, y)

    x1, y1, x2, y2 = best
    return (x1, y1), (x2, y2)


def y_on_line(p1, p2, x):
    """
    Compute y-value on the line passing through p1 and p2 at a given x (linear interpolation).
    """
    (x1, y1), (x2, y2) = p1, p2

    # Avoid division by zero for vertical-ish lines
    if x2 == x1:
        return (y1 + y2) / 2.0

    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def in_half(cx, cy, net_p1, net_p2, half="far"):
    """
    Decide whether a point (cx, cy) lies in the requested half-court,
    based on whether it is above or below the net line.

    Args:
        half: "far"  -> above the net (typically farther from camera)
              "near" -> below the net (typically closer to camera)
    """
    y_net = y_on_line(net_p1, net_p2, cx)
    return (cy < y_net) if half == "far" else (cy > y_net)


def main():
    ap = argparse.ArgumentParser(
        description="Half-court player detection using YOLO + net-line split + DET_EVERY/HOLD speed-up."
    )
    ap.add_argument("--video_in", required=True, help="Input video path")
    ap.add_argument("--video_out", default="out_player.mp4", help="Output annotated video path")
    ap.add_argument("--csv_out", default="player.csv", help="Output CSV path")

    ap.add_argument(
        "--half",
        choices=["far", "near"],
        default="far",
        help="far=above net (far side), near=below net (near side)"
    )

    ap.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path (.pt or OpenVINO export folder)")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold (NMS)")
    ap.add_argument("--max_det", type=int, default=10, help="Max detections per frame")
    ap.add_argument("--imgsz", type=int, default=512, help="Inference size (must match fixed-shape OpenVINO models)")

    # Speed trick: run YOLO every N frames and reuse (hold) last box in between
    ap.add_argument(
        "--det_every",
        type=int,
        default=10,
        help="Run YOLO every N frames, and HOLD last bbox in between (speed-up)."
    )
    ap.add_argument(
        "--max_hold",
        type=int,
        default=60,
        help="If no valid detection for this many frames, clear the held bbox."
    )

    # NOTE: tracking is only meaningful when running inference every frame (det_every=1)
    ap.add_argument(
        "--use_track",
        action="store_true",
        help="Use YOLO tracking (only effective when det_every=1)."
    )

    ap.add_argument("--show", action="store_true", help="Show visualization window (can reduce FPS)")
    ap.add_argument("--show_every", type=int, default=1, help="Show every N-th frame in the preview window")
    args = ap.parse_args()

    # -----------------------------
    # Video I/O
    # -----------------------------
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame from input video")

    H, W = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (W, H))

    # CSV output: one row per frame (box or empty), plus a "mode" field:
    # det  = ran YOLO and found a valid detection
    # hold = reused the last valid box
    # none = ran YOLO but found nothing valid
    csv_f = open(args.csv_out, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "track_id", "mode"])

    # -----------------------------
    # Model
    # -----------------------------
    model = YOLO(args.model)

    # Detect net line once on the first frame (fast, but assumes camera is fixed)
    net_p1, net_p2 = find_net_line(first)

    # -----------------------------
    # HOLD state (used when det_every > 1)
    # -----------------------------
    last_box = None
    last_conf = None
    last_id = None
    since_good = 0  # frames since the last valid detection

    # Restart video from frame 0 because we already consumed the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    # Simple heuristics to reject "weird" boxes (helps remove false positives)
    # Aspect ratio here is (h / w) because a standing person is taller than wide.
    AR_MIN, AR_MAX = 1.1, 8.0
    MIN_AREA = 0.00003 * (W * H)   # allow small far players
    MAX_AREA = 0.10 * (W * H)      # reject huge boxes (e.g., walls/net)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decide whether to run YOLO on this frame
        run_det = (frame_idx % max(1, args.det_every) == 0)
        mode = "hold"

        if run_det:
            # Tracking is only useful if we run inference every frame
            use_track_now = (args.use_track and args.det_every == 1)

            if use_track_now:
                res = model.track(
                    frame,
                    persist=True,
                    conf=args.conf,
                    iou=args.iou,
                    classes=[0],            # person class (COCO)
                    max_det=args.max_det,
                    imgsz=args.imgsz,
                    verbose=False
                )[0]
            else:
                res = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    classes=[0],            # person class (COCO)
                    max_det=args.max_det,
                    imgsz=args.imgsz,
                    verbose=False
                )[0]

            best = None
            best_score = -1

            # Parse detections (if any)
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                ids = None
                if use_track_now and res.boxes.id is not None:
                    ids = res.boxes.id.cpu().numpy().astype(int)

                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    c = float(confs[i])
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Box center
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Keep only the requested half-court
                    if not in_half(cx, cy, net_p1, net_p2, half=args.half):
                        continue

                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue

                    # Person-like box constraints
                    ar = h / (w + 1e-6)
                    area = w * h
                    if not (AR_MIN <= ar <= AR_MAX):
                        continue
                    if not (MIN_AREA <= area <= MAX_AREA):
                        continue

                    tid = int(ids[i]) if ids is not None else -1

                    # Choose best purely by confidence (simple & stable)
                    score = c
                    if score > best_score:
                        best_score = score
                        best = (x1, y1, x2, y2, c, tid)

            # Update HOLD state
            if best is not None:
                x1, y1, x2, y2, c, tid = best
                last_box = (x1, y1, x2, y2)
                last_conf = c
                last_id = tid if tid != -1 else last_id
                since_good = 0
                mode = "det"
            else:
                since_good += 1
                mode = "none"
        else:
            # Not running detection on this frame -> we are in HOLD mode
            since_good += 1

        # If we haven't seen a valid detection for too long, clear the held box
        if since_good >= args.max_hold:
            last_box = None
            last_conf = None
            last_id = None

        # -----------------------------
        # Visualization + outputs
        # -----------------------------
        vis = frame.copy()

        if last_box is not None:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            c_txt = f"{last_conf:.2f}" if last_conf is not None else "?"
            cv2.putText(
                vis,
                f"player {c_txt}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            writer.writerow([
                frame_idx, x1, y1, x2, y2,
                f"{(last_conf or 0):.4f}",
                (last_id if last_id is not None else ""),
                mode
            ])
        else:
            writer.writerow([frame_idx, "", "", "", "", "", "", mode])

        out.write(vis)

        # Optional live preview (can reduce FPS)
        if args.show and (frame_idx % max(1, args.show_every) == 0):
            cv2.imshow("Player Detection (DET_EVERY + HOLD)", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        frame_idx += 1

    # -----------------------------
    # Cleanup
    # -----------------------------
    cap.release()
    out.release()
    csv_f.close()
    cv2.destroyAllWindows()
    print(f"Done.\nVideo: {args.video_out}\nCSV: {args.csv_out}")


if __name__ == "__main__":
    main()
