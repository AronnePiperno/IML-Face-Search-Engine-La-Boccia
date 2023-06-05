import os
import cv2
from deepface.commons import functions
import torchvision

def face_alignment(input_dir: str, output_dir: str):
    for file_name in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        detection = functions.extract_faces(img=img, enforce_detection=False)
        x, y, w, h = detection[0][1].values()
        aligned_img = img[int(y):int(y + h), int(x):int(x + w)]
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGRA2RGB)
        aligned_img = cv2.resize(aligned_img, (160, 160))
        cv2.imwrite(os.path.join(output_dir, file_name), aligned_img)


def embeddings_calc(gallery_paths: list, model, device, data_dir: str) -> dict:
    embeddings = {}
    for path in gallery_paths:
        image = cv2.imread(os.path.join(data_dir, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Resize((224, 224), antialias=True)(image)
        image = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        embeddings[path] = model(image).detach().cpu().numpy()
    return embeddings

