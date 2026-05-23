#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# install all necessary packages
import openslide
import numpy as np
from PIL import Image
import os
import random
from pathlib import Path

# --- CONFIG ---
# Adjust to computer path of .ndpi image location
NDPI_PATH = "/Users/sarah/Downloads/Data/02.ndpi"
# Adjust to computer path of desired image tiles outpupt
OUTPUT_DIR = "/Users/sarah/Downloads/Data/final_tiles_500"

TILE_SIZE = 512 # Width and height (in pixels) of each square tile
LEVEL = 0 # Zoom level to read from; 0 = highest resolution
MIN_TISSUE_PCT = 0.20 # 20% threshold for "dark" pixels to count as tissue

INITIAL_SAMPLE = 1500   # first sampling
FINAL_SAMPLE = 500      # final keep

RANDOM_SEED = 42 # fixed seed for reproducible runs

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# --- LOAD SLIDE ---
slide = openslide.OpenSlide(NDPI_PATH) # open slide

w, h = slide.level_dimensions[LEVEL] # get pixel width and height at the chosen zoom level
downsample = slide.level_downsamples[LEVEL] 

print(f"Scanning level {LEVEL}: {w} x {h}")

# =========================
# STEP 1: FIND TISSUE TILES
# =========================
# walk every possible tile position across the slide and keep only tiles that contain tissue
tissue_coords = []
skipped_background = 0

for y in range(0, h, TILE_SIZE):
    for x in range(0, w, TILE_SIZE):
        loc = (int(x * downsample), int(y * downsample))
        size = (min(TILE_SIZE, w - x), min(TILE_SIZE, h - y))

        tile = slide.read_region(loc, LEVEL, size).convert("RGB")
        arr = np.array(tile)

        dark_pixels = np.sum(arr < 200) / arr.size

        if dark_pixels < MIN_TISSUE_PCT:
            skipped_background += 1
            continue

        tissue_coords.append((x, y))

print(f"Tissue tiles found: {len(tissue_coords)}")
print(f"Background skipped: {skipped_background}")

# =========================
# STEP 2: SAMPLE 1500
# =========================
# randomly pick up to INITIAL_SAMPLE tiles from all tissue tiles to reduce the workload of quality filtering
initial_n = min(INITIAL_SAMPLE, len(tissue_coords))
sampled_coords = random.sample(tissue_coords, initial_n)

print(f"Initial sample: {initial_n}")

# =========================
# STEP 3: QUALITY FILTERING
# =========================
# re-read each sampled tile and discard blurry, low-contrast, or stain-poor tiles
good_tiles = [] # store tiles that pass all quality checks
removed_quality = 0 # counter for tiles rejected at this stage

for (x, y) in sampled_coords:
    loc = (int(x * downsample), int(y * downsample))
    size = (min(TILE_SIZE, w - x), min(TILE_SIZE, h - y))

    tile = slide.read_region(loc, LEVEL, size).convert("RGB")
    arr = np.array(tile)

    # --- Sharpness ---
    gray = np.mean(arr, axis=2) 
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = np.mean(gx**2) + np.mean(gy**2)

    # --- Contrast ---
    contrast = np.std(gray)

    # --- Color variation ---
    color_var = np.std(arr[:, :, 0] - arr[:, :, 1])

    # --- Filtering ---
    if sharpness < 20 or contrast < 15 or color_var < 10:
        removed_quality += 1
        continue

    good_tiles.append((x, y)) # tile passed all checks

print(f"After quality filtering: {len(good_tiles)} kept")
print(f"Removed (quality): {removed_quality}")

# =========================
# STEP 4: FINAL SAMPLE
# =========================
# randomly draw the final set of tiles from the quality-filtered pool
final_n = min(FINAL_SAMPLE, len(good_tiles))
final_coords = random.sample(good_tiles, final_n)

print(f"Final sample size: {final_n}")

# =========================
# STEP 5: SAVE FINAL IMAGES
# =========================
# save all final chosen tiles in output directory
saved = 0

for (x, y) in final_coords:
    loc = (int(x * downsample), int(y * downsample))
    size = (min(TILE_SIZE, w - x), min(TILE_SIZE, h - y))

    tile = slide.read_region(loc, LEVEL, size).convert("RGB")
    # build output filename encoding tile's position
    out_path = os.path.join(
        OUTPUT_DIR, f"tile_{y:06d}_{x:06d}.jpg"
    )
    tile.save(out_path, "JPEG", quality=90)
    saved += 1
# print number of saved tiles and output directory
print("\n======================")
print(f"Saved final tiles: {saved}")
print(f"Output directory: {OUTPUT_DIR}")
print("======================")

slide.close()