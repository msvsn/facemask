import cv2 as cv
import mediapipe as mp
import numpy as np

face_images = {
    "Name": "path_to_your_image"
}
current_face_image = "path_to_your_image"

def load_face_image(path):
    face_img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if face_img is None:
        print(f"Error: Could not load image {path}")
        exit()
    if face_img.shape[2] != 4:
        print(f"Error: The image {path} does not have an alpha channel")
        exit()
    return face_img

other_face = load_face_image(current_face_image)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

other_face_rgb = cv.cvtColor(other_face[:, :, :3], cv.COLOR_BGR2RGB)
results_other = face_mesh.process(other_face_rgb)
if not results_other.multi_face_landmarks:
    print("No face landmarks detected on the other face image.")
    exit()

landmarks_other = results_other.multi_face_landmarks[0]
mesh_points_other = np.array([np.multiply([p.x, p.y], [other_face.shape[1], other_face.shape[0]]).astype(int) for p in landmarks_other.landmark])
other_face_mask = other_face[:, :, 3]

video_capture = cv.VideoCapture(0)
cv.namedWindow("moya")
face_mesh_video = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

labels = {
    "1": (10, 30),
    "2": (10, 60),
    "3": (10, 90),
    "4": (10, 120),
    "5": (10, 150),
    "6": (10, 210),
    "7": (10, 240),
    "Remove Mask": (10, 180)
}

def click_event(event, x, y, flags, param):
    global current_face_image, other_face, mesh_points_other, other_face_mask

    if event == cv.EVENT_LBUTTONDOWN:
        for label, pos in labels.items():
            lx, ly = pos
            if lx <= x <= lx + 100 and ly - 20 <= y <= ly:
                if label == "Remove Mask":
                    current_face_image = None
                else:
                    current_face_image = face_images[label]
                    other_face = load_face_image(current_face_image)
                    other_face_rgb = cv.cvtColor(other_face[:, :, :3], cv.COLOR_BGR2RGB)
                    results_other = face_mesh.process(other_face_rgb)
                    if not results_other.multi_face_landmarks:
                        print("No face landmarks detected on the other face image.")
                        return
                    landmarks_other = results_other.multi_face_landmarks[0]
                    mesh_points_other = np.array([np.multiply([p.x, p.y], [other_face.shape[1], other_face.shape[0]]).astype(int) for p in landmarks_other.landmark])
                    other_face_mask = other_face[:, :, 3]

cv.setMouseCallback("moya", click_event)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    height, width, _ = frame.shape

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh_video.process(rgb_frame)
    if results.multi_face_landmarks and current_face_image:
        for detection in results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in detection.landmark])

            points_user = np.array([
                mesh_points[10],
                mesh_points[234],
                mesh_points[454],
                mesh_points[152],
            ])
            points_other = np.array([
                mesh_points_other[10],
                mesh_points_other[234],
                mesh_points_other[454],
                mesh_points_other[152],
            ])

            M, _ = cv.estimateAffinePartial2D(points_other, points_user)
            transformed_other_face = cv.warpAffine(other_face, M, (width, height))
            transformed_other_mask = cv.warpAffine(other_face_mask, M, (width, height))

            alpha_mask = transformed_other_mask / 255.0
            for c in range(0, 3):
                frame[:, :, c] = frame[:, :, c] * (1 - alpha_mask) + transformed_other_face[:, :, c] * alpha_mask

    for label, pos in labels.items():
        lx, ly = pos
        cv.putText(frame, label, (lx, ly), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

    cv.imshow("moya", frame)
    if cv.waitKey(27) & 0xFF == 27:
        break

video_capture.release()
cv.destroyAllWindows()
