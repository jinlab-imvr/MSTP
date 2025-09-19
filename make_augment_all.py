import os
import random
import math
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def random_flip_and_rotate(image):
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    original_w, original_h = image.size
    angle = random.uniform(-10, 10)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    new_w, new_h = image.size
    rad = math.radians(abs(angle))
    sin_a, cos_a = math.sin(rad), math.cos(rad)
    crop_w = int(original_w * cos_a - original_h * sin_a)
    crop_h = int(original_h * cos_a - original_w * sin_a)
    crop_w = max(1, min(crop_w, new_w))
    crop_h = max(1, min(crop_h, new_h))

    left = (new_w - crop_w) / 2
    top = (new_h - crop_h) / 2
    right = left + crop_w
    bottom = top + crop_h

    image = image.crop((left, top, right, bottom))
    image = image.resize((original_w, original_h), resample=Image.BICUBIC)
    return image


def process_file(file_path, data_dir, save_dir, augmentation_times):
    try:
        with Image.open(file_path) as image:
            for i in range(augmentation_times):
                new_file_path = file_path.replace(data_dir, save_dir).replace(".jpg", f"_{i}.jpg")
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                new_image = random_flip_and_rotate(image)
                new_image.save(new_file_path)
    except Exception as e:
        return f"Error processing {file_path}: {e}"
    return None


if __name__ == "__main__":
    data_dir = "GraSP/train/frames"
    save_dir = "GraSP/augmented_train/frames"
    augmentation_times = 8

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(data_dir)
        for f in files if f.lower().endswith(".jpg")
    ]

    errors = []
    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, f, data_dir, save_dir, augmentation_times) for f in all_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            err = future.result()
            if err:
                errors.append(err)

    if errors:
        print("Some files failed:")
        for e in errors:
            print(e)
