import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

#640x640 crop
TILE_SIZE = 64
OUTPUT_SIZE =  int(640 * DOWNSCALE)
DOWNSCALE = 0.25
BATCH_SIZE = 128

# Sobel fiters
sobel_x = torch.tensor([[[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]], dtype=torch.float32, device=device)
sobel_y = torch.tensor([[[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]], dtype=torch.float32, device=device)

def batch_calc_tile_scores(tile_list):
    gray_tensors = []
    brightness_vals = []
    hue_stds = []

    for tile in tile_list:
        # HSV hue std
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:, :, 0])
        hue_stds.append(hue_std)

        # Brightness
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_vals.append(brightness)

        gray = gray.astype(np.float32) / 255.0
        gray_tensor = torch.from_numpy(gray).unsqueeze(0)  # [1, H, W]
        gray_tensors.append(gray_tensor)

    gray_batch = torch.stack(gray_tensors).unsqueeze(1).to(device)  # [B, 1, H, W]

    edge_x = F.conv2d(gray_batch, sobel_x.unsqueeze(0), padding=1)
    edge_y = F.conv2d(gray_batch, sobel_y.unsqueeze(0), padding=1)
    edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edge_density = edge_mag.mean(dim=[1, 2, 3]).cpu().numpy()  # [B]

    brightness_vals = np.array(brightness_vals)
    hue_stds = np.array(hue_stds)

    scores = (brightness_vals * 0.4) + (edge_density * 50 * 0.4) + (hue_stds * 2 * 0.2)
    return scores.tolist()

def map_tile_to_original_coords(x_small, y_small, tile_w, tile_h, scale_x, scale_y):
    x_orig = int(x_small / scale_x)
    y_orig = int(y_small / scale_y)
    w_orig = int(tile_w / scale_x)
    h_orig = int(tile_h / scale_y)
    return x_orig, y_orig, w_orig, h_orig

def process_image(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"HATA: {img_path} okunamadı")
        return

    h, w = img.shape[:2]
    small_img = cv2.resize(img, (int(w * DOWNSCALE), int(h * DOWNSCALE)))
    small_h, small_w = small_img.shape[:2]

    tile_size = TILE_SIZE
    positions = []
    tile_list = []

    for y in range(0, small_h - tile_size + 1, tile_size):
        for x in range(0, small_w - tile_size + 1, tile_size):
            tile = small_img[y:y + tile_size, x:x + tile_size]
            if np.mean(tile) < 10:  # boşluk filtresi
                continue
            positions.append((x, y))
            tile_list.append(tile)

    # Batching-score
    all_scores = []
    for i in range(0, len(tile_list), BATCH_SIZE):
        batch_tiles = tile_list[i:i + BATCH_SIZE]
        batch_scores = batch_calc_tile_scores(batch_tiles)
        all_scores.extend(batch_scores)

    # Score mapping
    score_map = np.zeros((small_h // tile_size, small_w // tile_size))
    for (x, y), score in zip(positions, all_scores):
        ix = y // tile_size
        jx = x // tile_size
        score_map[ix, jx] = score

    tiles_per_row = OUTPUT_SIZE // TILE_SIZE

    max_score = -1
    max_pos = (0, 0)
    for i in range(score_map.shape[0] - tiles_per_row + 1):
        for j in range(score_map.shape[1] - tiles_per_row + 1):
            window_score = np.sum(score_map[i:i + tiles_per_row, j:j + tiles_per_row])
            if window_score > max_score:
                max_score = window_score
                max_pos = (i, j)

    x_small = max_pos[1] * tile_size
    y_small = max_pos[0] * tile_size

    scale_x = DOWNSCALE
    scale_y = DOWNSCALE
    x_orig, y_orig, w_orig, h_orig = map_tile_to_original_coords(
        x_small, y_small, OUTPUT_SIZE, OUTPUT_SIZE, scale_x, scale_y
    )

    crop_img = img[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f"cropped_{filename}")
    cv2.imwrite(output_path, crop_img)

    print(f"{filename} done and saved {output_path}")

def process_images(input_folder, output_folder):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        img_path = os.path.join(input_folder, f)
        process_image(img_path, output_folder)

if __name__ == "__main__":
    input_folder = r"INPUT_PATH"
    output_folder = r"OUTPUT_PATH"
    process_images(input_folder, output_folder)
