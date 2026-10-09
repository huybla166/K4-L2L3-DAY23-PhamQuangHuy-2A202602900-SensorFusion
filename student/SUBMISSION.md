# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phạm Quang Huy
- MSSV: 2A202602900
- Email: huybla166@gmail.com
- Link repo (fork): https://github.com/huybla166/K4-L2L3-DAY23-PhamQuangHuy-2A202602900-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): commit cuối trên `main`, nộp kèm trên LMS (artifacts chấm điểm được sinh và commit ở `ce3434f95ae83c50032d4e07d54bf34e14b10ccf`, CP5; code E–H không đổi sau commit đó)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, seed `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: **0.1503 m**, 502, 11.3437 m², 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: **0.1359 m**, 502, 9.2668 m², 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: xem bảng và các ý bên dưới.

| Chỉ số | lidar | fused | Ngưỡng đủ điểm |
|---|---|---|---|
| RMSE 3D (m) | 0.1503 | 0.1359 | ≤ 0.45 |
| `matches` / `ghost_track_frames` / `missed_gt_frames` | 502 / 0 / 239 | 502 / 0 / 239 | — |
| `precision_track` = 502 / (502 + 0) | 1.000 | 1.000 | ≥ 0.75 |
| `coverage` = 502 / `det_tp` 519 | 0.967 | 0.967 | ≥ 0.70 |
| `rmse_fused − rmse_lidar` | | −0.0145 m | ≤ +0.05 m |

- **Hai mode ghép đúng cùng các cặp track–GT.** Ngoài `metrics.json` (matches, ghost, miss, mean confirmed bằng nhau), trong `grade_run.log` các trường `det_tp/det_fp/det_fn/valid_gt` **và** `confirmed/matches/ghosts/misses` trùng nhau ở cả 199 frame; chỉ `sum_sq_err` khác (195/199 frame). Vì vậy so sánh RMSE là công bằng: cùng 502 cặp, fused giảm tổng bình phương sai số từ 11.34 xuống 9.27 m² (−18 %), RMSE −0.0145 m.
- **RMSE thấp không phải nhờ bỏ bớt track khó:** không có ghost (`precision_track` = 1), và `coverage` 0.967. `missed_gt_frames` = 239 = 222 (`det_fn`, detector bỏ sót, tracker không có đo để theo) + 17 (xe đã được detect nhưng track chưa confirmed): 4 frame đầu của mỗi track mới (cần 5 hit, ví dụ frame 0–3 có `valid_gt` = 2, `confirmed` = 0; frame 4 `confirmed` = 2) và các lần track bị tạo lại (mục Bonus CVAT).
- **Camera tác động theo trục nào** (đo trên 484 lần camera update của track đã ghép GT, `student/bonus/viz/viz_summary.json`): |sai số| trung bình x 0.087 → 0.074 m, y 0.061 → 0.050 m nhưng z 0.051 → 0.062 m. Camera đo hướng nhìn (u ↔ y/x, v ↔ z/x), nên kéo vị trí ngang tốt; z bị lệch vì tâm hộp 2D không phải hình chiếu tâm hộp 3D (xem câu 4 và giới hạn bên dưới).
- **Giới hạn của đo camera mô phỏng:** đo camera = tâm hộp 2D **ground-truth** FRONT + nhiễu N(0, 0.5 px) theo seed, trong khi EKF giả định σ = 5 px. Không có miss, FP hay nhầm lớp như một camera detector thật, nên fused tốt hơn ở đây không chứng minh camera detector tốt. Ngược lại, dù là GT, đo này vẫn có **bias** so với mô hình h(x): innovation trung bình ở calibration gốc là (−9.5, +8.5) px (bảng ở mục Bonus), vì hộp 2D bao mặt nhìn thấy của xe, không phải tâm 3D, và camera/lidar không chụp cùng lúc tuyệt đối.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   - *Lidar:* `z = [x, y, z]ᵀ` (m), tâm hộp 3D của detector; trong lab lidar trùng hệ xe (`veh_to_sens = I`). `R = diag(0.1², 0.1², 0.1²)` m². `h(x)` tuyến tính (`R_vs·p + t`), nên `H = [R_vs | 0]` (3×6) là hằng số. `dim_meas = 3`, cổng χ²(0.995, 3) = 12.84.
   - *Camera:* `z = [u, v]ᵀ` (pixel), tâm hộp 2D FRONT. `R = diag(5², 5²)` px² (`build_camera_measurement` trong `camera_fusion.py`). `h(x)` phi tuyến pinhole: `p_s = R p + t`, `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (`camera_measurement_prediction`). Vì vậy `H` là Jacobian 2×6 tính lại tại mỗi `x` (platform `get_H`), cột vận tốc bằng 0. `dim_meas = 2`, cổng χ²(0.995, 2) = 10.60.
   - *Hệ quả:* camera chỉ đo **hướng** (một pixel ứng với cả một tia), gần như không có thông tin depth x. Độ nhạy `∂u/∂y = −f_i/x_s` ≈ −2083/25 ≈ −83 px/m ở 25 m, nên σ = 5 px tương đương khoảng 6 cm ngang. `ekf_update` trong `kalman.py` dùng chung một công thức `K = P Hᵀ S⁻¹`; hai sensor chỉ khác `meas.z`, `meas.R`, `get_hx`, `get_H`. Vì đơn vị khác nhau (m và px), innovation chỉ so sánh được sau khi chuẩn hoá bằng `S`.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   - `d² = γᵀ S⁻¹ γ` với `S = H P Hᵀ + R` gộp cả độ bất định của track lẫn nhiễu cảm biến. Nếu mô hình đúng, `d² ~ χ²(dim_meas)`, nên ngưỡng `chi2.ppf(0.995, dim)` loại các cặp có xác suất dưới 0.5 % (`association_cost_matrix` gán `inf`). Không có cổng, greedy `pick_next_pair` sẽ luôn ghép cặp gần nhất kể cả khi vô lý (FP detection, xe bên cạnh), kéo track lệch và sinh ghost hoặc đổi ID. Đo bị loại thì nằm trong `unassigned_meas`, và lượt lidar khởi tạo track mới cho nó.
   - *Khác Euclidean khi `P` lớn:* track mới có σ_v = 50 m/s, nên sau một predict `P_xx = 0.01 + 0.1²·2500 + 0.3 = 25.3` m² (σ ≈ 5 m) và cổng 3D rộng `√(12.84·25.3)` ≈ 18 m. Track confirmed trong lần chạy này có `P_xx` sau predict khoảng 0.34 m² (trung vị), nên cổng chỉ còn ≈ `√(12.84·0.35)` ≈ 2.1 m. Một detection lệch 3 m được nhận cho track mới (d² ≈ 0.36) nhưng bị loại cho track confirmed (d² ≈ 26). Ngưỡng Euclidean cố định không phân biệt được hai trường hợp, và cũng không so được m với px hay theo hướng elip của `S`.
   - Thứ tự trong code: `association_cost_matrix` kiểm tra `meas.sensor.in_fov(track.x)` **trước**, nên cặp ngoài FOV hoặc depth ≤ 1e-6 không bao giờ được chiếu (`test_out_of_fov_gating_precedes_projection`). Sau đó mới tính `mahalanobis_distance` và `chi2_gate`.

3. **Track-then-fuse hay fuse-then-track? Chỉ ra trên log.**
   - **Track-then-fuse.** Trong `run_lab.py`, mỗi frame: detection lidar (BEV → FPN) → `KF.predict` mọi track **một lần** → `associate_and_update(manager, lidar_obs, KF, lidar_sensor)` (AssocL: update EKF + `manage_tracks` lidar) → `associate_and_update(manager, camera_obs, KF, camera_sensor)` (AssocC: chỉ update EKF; `manage_tracks` với camera return ngay). Không có bước nào trộn dữ liệu thô trước detection.
   - Log `grade_run.log`, frame 88:
     `lidar {det_tp 3, det_fp 0, det_fn 0, valid_gt 3, confirmed 2, matches 2, ghosts 0, misses 1, sum_sq_err 0.0661}`
     `fused {det_tp 3, det_fp 0, det_fn 0, valid_gt 3, confirmed 2, matches 2, ghosts 0, misses 1, sum_sq_err 0.0251}`.
     Trên cả 199 frame, mọi trường detection và vòng đời giống hệt nhau, chỉ `sum_sq_err` khác. Detection không bị camera ảnh hưởng (nếu fuse-then-track thì `det_*` sẽ khác giữa hai mode); camera chỉ tinh chỉnh trạng thái của track đã có. Hình `student/bonus/viz/01_*.png` và `02_*.png` cho thấy trạng thái sau AssocL (tam giác/vòng tròn) và sau AssocC (dấu +) trên cùng một track.

4. **Camera lệch calibration → triệu chứng trên innovation/residual?** (đo thật, bonus calibration)
   - **Innovation có bias hệ thống** (trung bình ≠ 0, cùng dấu qua nhiều frame và nhiều xe). Lệch yaw δ dời mọi điểm khoảng `f_i·tan δ` ≈ 36 px/độ theo u (`f_i` = 2083 px). Đo được γ_u trung bình (ứng viên gần nhất): −9.5 px (gốc) → −18.4 (0.25°) → −27.8 (0.5°) → −42.1 (1°) → −61.6 px (2°). Lệch pitch dời v (γ_v +8.5 → +29.3 px ở 1°). Lệch tịnh tiến 0.3 m dời u một lượng **phụ thuộc depth** (`f·Δy/x`, γ_u −34.6 px).
   - **NIS tăng vượt số bậc tự do:** NIS trung bình của các update được nhận là 2.25 ở calibration gốc (≈ 2 = dim_meas, filter nhất quán), 4.5 ở 0.25° và 7.3 ở 0.5°. **Tỉ lệ qua cổng χ² giảm:** 98.8 % → 93.9 % → 62.1 % → 2.6 % (1°).
   - **Residual sau update không về 0** và track bị kéo ngang mỗi frame, rồi lidar kéo lại ở frame sau ("giằng co"), nên RMSE fused tăng từ 0.136 lên 0.173 rồi 0.211 m (0.5°). Lệch nhỏ (bias < bán kính cổng ≈ 30–40 px) lọt qua cổng và làm hỏng trạng thái. Lệch lớn (≥ 1°) bị gating chặn gần hết, camera gần như bị tắt: RMSE về gần lidar (5°: 0.1503 m = lidar).

5. **Vì sao cần `sensor` tường minh ở frame rỗng; vì sao lidar quyết định score/init/delete?**
   - Khi `meas_list` rỗng, không thể suy ra sensor từ `meas_list[0].sensor`. Nhưng lượt **lidar** rỗng vẫn phải chạy `manage_tracks(unassigned_tracks, [], lidar)`: mọi track trong FOV lidar bị trừ `1/window`, và track hết score hoặc có `P` quá lớn bị xoá. Nếu bỏ bước này, frame không có detection sẽ "đóng băng" vòng đời và ghost sống mãi (`test_empty_lidar_frame_scores_then_deletes_exhausted_track`). Ngược lại, một lượt **camera** rỗng (hoặc frame không có nhóm FRONT) không được tính là miss. `TrackManager` phân biệt hai trường hợp chỉ nhờ `sensor.name`, nên `associate_and_update` luôn kết thúc bằng `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`.
   - Lidar quyết định tồn tại vì bốn lý do:
     - Lidar đo 3D đầy đủ nên khởi tạo được `x`; một pixel camera không có depth nên không init được track 3D.
     - FOV của FRONT chỉ khoảng ±24.7° (suy từ intrinsics: `atan((c_i−W)/f_i)` = −24.8°, `atan(c_i/f_i)` = +24.7°) và hay bị che khuất. Nếu camera được trừ score, các track ngoài ảnh sẽ bị phạt oan.
     - `window` = 6 được định nghĩa là 6 lượt lidar (0.6 s); camera cộng/trừ thêm sẽ đếm đôi.
     - Camera của lab lấy từ nhãn GT; nếu nó quyết định tồn tại thì GT rò vào lifecycle và hai mode không còn so sánh công bằng.

     Vì vậy, trong code, camera chỉ gọi `filter_obj.update`; `handle_updated_track` bỏ qua sensor không phải lidar, và `manage_tracks` return sớm. Log xác nhận điều này: `confirmed/matches/ghosts/misses` của hai mode trùng nhau ở mọi frame.

6. **Điều kiện xác nhận, giữ confirmed sau miss, xoá track** (`track_management.py`).
   - *Khởi tạo:* mỗi đo lidar không ghép được tạo một track với `x = [sens_to_veh·z; 0, 0, 0]`, `P_pos = R_rot·R·R_rotᵀ` (0.01 m²), `P_vel = diag(50², 50², 5²)`, `score = 1/6`, `state = initialized`.
   - *Hit lidar:* `score = min(1, score + 1/6)`; chưa đạt ngưỡng thì `tentative`. **Xác nhận khi `score > 0.8`**, tức sau 5 lần đo liên tiếp (1/6 → 5/6 = 0.833). Ví dụ: track #5 khởi tạo ở frame 44 và confirmed ở frame 48; track #10 khởi tạo ở 92 và confirmed ở 98.
   - *Miss trong FOV lidar:* `score −= 1/6` (track ngoài FOV không bị trừ). **Track confirmed không bị hạ trạng thái** vì một miss: 1.0 → 0.833 → 0.667 vẫn confirmed (`test_lidar_score_caps_and_confirmation_survives_one_miss`).
   - *Xoá (điều kiện OR):* `P[0,0]` hoặc `P[1,1]` > `max_P` = 9 m² (σ > 3 m), bất kể score; confirmed và `score < 0.6` (đúng biên 0.6 thì giữ); chưa confirmed và `score ≤ 0`. Ví dụ thật: track #5 ở frame 58 có score 0.667, miss tiếp ở frame 59 → 0.5 < 0.6 → bị xoá; ghost #4 (từ FP detection) lên 0.5 rồi rơi về 0 ở frame 33 → bị xoá mà chưa bao giờ confirmed.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- **Cách tạo lại mọi số liệu bonus:** `student/bonus/bonus_analysis.py`. Stage `cache` chạy detector có sẵn một lần (lưu vào `data/cache/`, không commit). Các stage khác chạy lại **đúng vòng lặp** `run_lab.run` (cùng `Sensor`, `Filter`, `TrackManager`, `evaluation`, cùng workspace E–H) trên cache đó. Stage `verify` xác nhận replay cho kết quả **giống hệt từng bit** với `student/artifacts/metrics_lidar.json` và `metrics_fused.json`. Không stage nào ghi vào `student/artifacts/`; artifacts chấm điểm là lần chạy `--fusion compare --seed 0` gốc.

- **Bonus 1 — Export CVAT (+3).** `fusion_lab.export_cvat.export_tracks_json` ghi `student/bonus/cvat/tracks_fused.json` (mọi track, mọi frame: id, state, score, x/y/z, v, kích thước, yaw, GT được ghép, cờ ghost). Từ đó tạo `student/bonus/cvat/annotations_cvat_video_1.1.xml` (định dạng *CVAT for video 1.1*: label `fusion_track` với thuộc tính `track_id`/`state`/`ghost`, và label `gt_vehicle`) trên chuỗi 199 ảnh BEV (xoay để hướng đi lên trên). Danh sách sự kiện ở `student/bonus/cvat/cvat_events.json`. XML được import vào CVAT v2.78 chạy local (định dạng "CVAT 1.1", qua REST API) bằng `student/bonus/cvat_import.py`; script này cũng chụp ảnh giao diện annotation ở các frame sự kiện: `student/bonus/cvat/cvat_ghost_track4_frame30.png`, `cvat_id_change_5_to_7.png`, `cvat_id_change_8_9_10.png`.
  - *Cách đọc ảnh:* số object của CVAT = `track_id` + 1 (track #5 là object 6). Ảnh ghost bật "luôn hiện chi tiết" cho mọi object. Ở hai ảnh đổi ID, hộp track và hộp GT gần trùng nhau nên nhãn chữ che nhau; vì vậy chỉ track được trỏ chuột mới hiện nhãn trên canvas, và thanh bên mở mục DETAILS của track (`track_id`, `state`) và của xe GT (`gt_id`).
  - *Ghost:* track **#4** (frame 26–33) được tạo từ FP detection, không có GT nào trong cổng 2 m. Nó lên `tentative` (score 0.5) rồi bị xoá ở frame 33 khi score về 0, nên **không bao giờ confirmed**. Đó là lý do `ghost_track_frames = 0` (metric chỉ đếm track confirmed): ngưỡng xác nhận 5/6 lọc được ghost này. Ghost tương tự: #6 (44–49), #14 (143–148).
  - *Đổi ID:* xe GT `VJ3F-NiH…` được theo bởi track **#5** (confirmed từ frame 48). Xe này ở x ≈ 49.5 m, sát mép cửa sổ BEV 50 m, nên detector bỏ sót ở frame 55, 56, 58 và 59. Score của #5 đi 1.0 → 0.833 → 0.667 → (hit ở 57) 0.833 → 0.667 → 0.5 < 0.6, nên #5 bị xoá ở frame 59. Đến frame 60 xe được detect lại và khởi tạo thành track **#7** mới; cùng một xe đổi ID từ 5 sang 7. Trên `cvat_id_change_5_to_7.png`: frame 58 có `track_id 5` (confirmed), frame 60 có `track_id 7` (initialized), cả hai nằm trên cùng object GT 23 với `gt_id VJ3F-NiHmRQh0umylCVJmA`. Xe `8EFRSwEX…` cũng bị tạo lại 8 → 9 → 10 (frame 85–92) trước khi #10 ổn định (`cvat_id_change_8_9_10.png`: `track_id` 8, 9, 10 đều trên object GT 17, `gt_id 8EFRSwEXBf9P-SzIDRzx0A`). Đây là giới hạn của quản lý track chỉ dựa trên lidar, không có re-identification.

- **Bonus 2 — Trực quan hoá track và đo (+3).** Ảnh trong `student/bonus/viz/`:
  - `01_bev_tracks_lidar_vs_fused.png`: BEV frame 88 với hộp và tâm GT (xanh lá), detection (x đỏ), track lidar-only (o xanh), track fused sau lidar update (tam giác tím) và sau camera update (dấu + cam), kèm zoom ±0.6 m. Với track **F0**, camera update dời Δy = +0.241 m về phía tâm GT: lệch ngang so với GT giảm từ khoảng 0.18 m xuống còn khoảng 0.07 m.
  - `02_front_projection_before_after_camera.png`: ảnh FRONT frame 88 với đo camera (ô vàng), `h(x)` sau lidar update (o xanh) và sau camera update (+ cam). Ở F0, innovation γ = (−31.7, +11.0) px giảm còn residual (−6.2, +2.1) px sau update; ở F1, γ = (−9.5, −2.0) px giảm còn (−2.7, −0.6) px.
  - `03_track_error_timeseries.png`: sai số theo từng trục của xe dẫn đầu qua 195 frame, lidar so với fused. Fused giảm sai số x và bỏ được đoạn lệch ngang lớn của lidar (đến −0.18 m quanh frame 80–90). Đổi lại, y chuyển sang bias dương khoảng +0.08 m và z bị đẩy xuống khoảng −0.10 m, cho thấy rõ bias của đo camera lấy từ nhãn 2D.
  - `04_camera_update_error_histogram.png`: trên **cùng** 484 lần update, sai số trước và sau camera update theo x/y/z. RMS 3D giảm từ 0.145 xuống 0.132 m; |e_x| 0.087 → 0.074, |e_y| 0.061 → 0.050, |e_z| 0.051 → 0.062 m.

- **Bonus 3 — Phân tích calibration (+4).** Tracker dùng extrinsic camera bị làm lệch (`perturbed_extrinsic`), trong khi đo camera vẫn sinh từ calibration **đúng**. Số liệu đầy đủ ở `student/bonus/calibration_results.json`; đồ thị ở `student/bonus/viz/05_calibration_yaw_sweep.png`. Lidar-only RMSE = 0.1503 m.

  | Mức lệch | γ_u TB (px) | γ_v TB (px) | Qua cổng χ² | NIS TB (update được nhận) | Số camera update | RMSE fused (m) | Δ so với lidar (m) |
  |---|---|---|---|---|---|---|---|
  | Gốc (0) | −9.5 | +8.5 | 98.8 % | 2.25 | 509 | 0.1359 | −0.014 |
  | yaw +0.25° | −18.4 | +8.6 | 93.9 % | 4.55 | 486 | 0.1732 | +0.023 |
  | yaw +0.5° | −27.8 | +8.8 | 62.1 % | 7.28 | 322 | **0.2112** | **+0.061** |
  | yaw +1° | −42.1 | +5.8 | 2.6 % | 3.51 | 20 | 0.1726 | +0.022 |
  | yaw +2° | −61.6 | −8.5 | 2.4 % | 3.92 | 19 | 0.1760 | +0.026 |
  | yaw +5° | −26.2* | −53.9* | 3.0 % | 2.04 | 28 | 0.1503 | −0.000 |
  | pitch +1° | −4.7 | +29.3 | 15.1 % | 5.08 | 87 | 0.1876 | +0.037 |
  | lateral +0.3 m | −34.6 | +8.7 | 32.7 % | 7.12 | 180 | 0.1847 | +0.034 |

  γ TB = innovation trung bình của ứng viên gần nhất với mỗi track confirmed trong FOV, trước gating. (*) Ở 5°, ứng viên gần nhất thường là **xe khác**, nên trung bình không còn phản ánh bias.

  Nhận xét:
  - **Triệu chứng:** bias của γ tăng gần tuyến tính theo độ lệch (≈ `f_i·tan δ`, ~9 px mỗi 0.25°). NIS vượt 2 (số bậc tự do) và tỉ lệ qua cổng giảm. Ngay cả ở calibration gốc, γ đã có bias (−9.5, +8.5) px, đó là bias của đo từ nhãn 2D.
  - **Vì sao gating chặn hoặc không chặn:** bán kính cổng theo pixel là `√(10.6·S_uu)`, với `S_uu = (f/x)²·P_yy + 25`. Track confirmed ở 20–30 m có σ khoảng 10–12 px, nên cổng khoảng 30–40 px. Gating chỉ chặn được outlier **lớn hơn cổng**. Bias nhỏ (0.25–0.5°, γ khoảng 18–28 px) vẫn lọt qua, vì cổng không phân biệt được bias hệ thống với nhiễu ngẫu nhiên, và làm RMSE xấu nhất: 0.5° vượt giới hạn +0.05 m của rubric. Khi bias vượt cổng (≥ 1°), gần như mọi camera update bị loại và camera bị tắt, nên RMSE quay về mức lidar (5°: 0.1503). Ở 1–2° vẫn còn khoảng 20 update lọt qua, chủ yếu là xe gần hơn (depth trung vị 19.9 m so với 27.2 m của mọi ứng viên), nơi `(f/x)²·P` và do đó cổng theo pixel lớn hơn; vì vậy RMSE vẫn cao hơn lidar khoảng 0.02 m.
  - **Kết luận:** quan hệ giữa lệch calibration và RMSE **không đơn điệu**. Lệch nhỏ nguy hiểm hơn lệch lớn, vì nó lọt qua cổng. Cần theo dõi trung bình innovation và NIS theo thời gian để phát hiện lệch calibration, thay vì dựa vào gating.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude Opus 5.5, Anthropic), dùng như agent trong terminal của repo
- Dùng cho phần nào (hàm, câu hỏi, debug): đọc hướng dẫn trong repo; viết code Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`) theo các dòng `# vi: TODO`; viết script bonus `student/bonus/bonus_analysis.py` (cache, replay, hình, sweep calibration, export CVAT); dựng CVAT local để import và chụp ảnh; soạn bản nháp báo cáo này từ số liệu đã chạy. Không sửa `platform/`, không sửa test, không sửa tay artifacts.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` → 128 passed (không còn failed/xfailed), và chạy riêng từng nhóm test theo CP1–CP4 (trên Windows cần `PYTHONUTF8=1`: nếu không, 6 test của `test_check_submission.py` lỗi ở bước setup vì `write_text` ghi tiếng Việt bằng encoding mặc định cp1252, không liên quan code E–H); chạy `fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0` trên frame 0–198. Mọi số trong báo cáo lấy từ `metrics.json`, `grade_run.log` và các file JSON do script bonus sinh ra; replay bonus được kiểm tra giống hệt từng bit với artifacts chấm điểm. Đối chiếu công thức F, Q, K, (I − KH)P, pinhole h(x), χ² và luật score/xoá với `docs/HUONG_DAN_KY_THUAT.md` và với ví dụ số trong test (ví dụ `P_pred` với dt = 0.2, q = 3). `python tools/check_submission.py` báo `SẴN SÀNG NỘP`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa) — không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
