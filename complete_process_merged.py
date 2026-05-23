# imports 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
#from cellpose import models
from cellpose_omni import models
import supervision as sv

# standard library
from collections import Counter
import requests
import os
import time
import shutil



#importing own functions
from functions import process_slide

# ================= TOTAL TIMER START =================
program_start = time.perf_counter()
# =====================================================

#****************************************************************************#
"""
To configure for your file system:
------
REPORT_PATH     : path to the *file* where all summary results will be printing to
INDV_REP_PATH   : path to the *file* where the meta data collected from each image 
                  in the INPUT_FOLDER will be printed to
INPUT_FOLDER    : path to the *folder* where the slide images are held
my_key          : the Roboflow key found under 
                  "Workspace Settings" --> "API Keys" --> "Private API Key"
                  in Roboflow online.
use_gpu         : boolean to use a GPU to accelerate the segmentation process. 
                  Note -- without it, this will take hrs depending on images processed.
use_omni        : boolean to use the Omnipose algorithm for non-morphological segmentation.
"""
# Summary Report
REPORT_PATH = Path(r"report.txt")

# Indiviudal Image Report
INDV_REP_PATH = Path(r"indv_report.txt")

# Input
INPUT_FOLDER = Path(r"Your_Image_Folder")

# Verify Input
# Check folder existence
if not INPUT_FOLDER.exists():
    print(f"[ERROR] INPUT_FOLDER does not exist: {INPUT_FOLDER}", file=rep_file)
    print("Exiting program.", file=rep_file)
    rep_file.close()
    indv_rep_file.close()
    exit()

# Check input is directory
if not INPUT_FOLDER.is_dir():
    print(f"[ERROR] INPUT_FOLDER is not a directory: {INPUT_FOLDER}", file=rep_file)
    print("Exiting program.", file=rep_file)
    rep_file.close()
    indv_rep_file.close()
    exit()

# Roboflow key
my_key     = "YOUR_KEY"

# Using gpu?
use_gpu = True

# Using omnipose?
use_omni = True


#*****************************************************************************#
"""
Removing output files from previous runs if needed
"""
# Delete __pycache__ directories
for pycache in Path(".").rglob("__pycache__"):
    try:
        shutil.rmtree(pycache)
        print(f"Deleted: {pycache}")
    except Exception as e:
        print(f"Could not delete {pycache}: {e}")

# Delete Batch Extracted WBCs folder if it exists
BATCH_OUTPUT_DIR = Path(r"Batch_Extracted_WBCs")

if BATCH_OUTPUT_DIR.exists() and BATCH_OUTPUT_DIR.is_dir():
    try:
        shutil.rmtree(BATCH_OUTPUT_DIR)
        print(f"Deleted output folder: {BATCH_OUTPUT_DIR}")
    except Exception as e:
        print(f"Could not delete {BATCH_OUTPUT_DIR}: {e}")


#*****************************************************************************#
"""
Setting up the output files for segmentation and classification results
"""
# Report files
rep_file = open(REPORT_PATH, "w")
indv_rep_file = open(INDV_REP_PATH, "w")

# Output
BATCH_OUTPUT_DIR = Path(r"Batch_Extracted_WBCs")

# Create output directory if it doesn't exist
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all image files
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(INPUT_FOLDER.glob(ext))

print("********     Set-up      ********", file=rep_file)
print(f"Input Folder:    {INPUT_FOLDER}", file=rep_file)
print(f"Output Folder for Segmented Images:  {BATCH_OUTPUT_DIR}", file=rep_file)
print(f"Found {len(image_files)} images to process\n", file=rep_file)

#*****************************************************************************#
"""
Initialize Omnipose or Cellpose model and run!
""" 
# Loading the Omnipose model
cp_model = models.CellposeModel(gpu=use_gpu, model_type="cyto2", omni=use_omni)
# Affirm successful loading of segmentation models and enabling of GPU
print("\nCellpose model loaded successfully", file=rep_file)
print(f"GPU enabled: {cp_model.gpu}", file=rep_file)
print(f"Omnipose enabled: {cp_model.omni}", file = rep_file)


# Running through all images in the given folder
tic = time.perf_counter()
all_results = []
for img_path in image_files:
    try:
        result = process_slide(img_path, BATCH_OUTPUT_DIR, cp_model, indv_rep_file)
        all_results.append(result)
    except Exception as e:
        print(f"ERROR processing {img_path.name}: {e}")
        print("\n" + "*"*60 + "\n")
toc = time.perf_counter()

# Overall summary
print("\n" + "="*60, file=rep_file)
print("BATCH PROCESSING COMPLETE", file=rep_file)
print("="*60, file=rep_file)
print(f"Images processed:      {len(all_results)}", file=rep_file)
print("Segmentation Time:     " + str(round(toc-tic, 3)) + " seconds", file=rep_file)
print(f"Total WBCs extracted:  {sum(r['wbcs_extracted'] for r in all_results)}", file=rep_file)
print(f"Total cells detected:  {sum(r['total_cells'] for r in all_results)}", file=rep_file)

#*****************************************************************************#
"""
Classifying each segemented image.
"""
print("\n" + "="*60, file=rep_file)
print("Classification Results", file=rep_file)
print("="*60, file=rep_file)

model_name = "aml-detection-fnqx1/16" # adjust model version

images = os.listdir(BATCH_OUTPUT_DIR)
images = [BATCH_OUTPUT_DIR/f for f in images]

# initializing counters
cell_counter = {"Basophil": 0, "Neutrophil": 0, "Erythroblast": 0, "Eosinophil": 0, "Myeloblast": 0, "WBC Other": 0, "RBC": 0}

# begin counting!
slide_counts = {}  # track per-slide counts

# initialize tracking lists
myeloblast_percentages = []
blast_percents = []
wbc_counts = []
total_cell_counts = []
slide_names = []

for img_pth in images:
    with open(img_pth, "rb") as f:
        img_bytes = f.read()
    # Requesting Roboflow model
    url = f"https://detect.roboflow.com/{model_name}?api_key={my_key}"
    results = requests.post(url, files={"file": img_bytes})

    detections = sv.Detections.from_inference(results)
    class_names = detections.data.get("class_name", [])
    class_counts = Counter(class_names)

    total_cells = sum(class_counts.values())
    for cls, cnt in class_counts.most_common():
        cell_counter[cls] += cnt

    # accumulate per original slide
    slide_name = "_".join(img_pth.stem.split("_")[:-2])  # strips _WBC_XXXX
    if slide_name not in slide_counts:
        slide_counts[slide_name] = {"Basophil": 0, "Neutrophil": 0, "Erythroblast": 0, "Eosinophil": 0, "Myeloblast": 0, "WBC Other": 0, "RBC": 0}
    for cls in ["Basophil", "Neutrophil", "Erythroblast", "Eosinophil", "Myeloblast", "WBC Other", "RBC"]:
        slide_counts[slide_name][cls] += class_counts.get(cls, 0)

# print per-slide ratios to indv_report
for slide_name, counts in slide_counts.items():
    m = counts["Myeloblast"]
    b = counts["Basophil"]
    n = counts["Neutrophil"]
    er = counts["Erythroblast"]
    eo = counts["Eosinophil"]
    wo = counts["WBC Other"]
    r = counts["RBC"]

    total_wbc = m + b + n + eo + wo + er
    total_cells = total_wbc + r

    if total_wbc > 0:
        pct = 100 * m / total_wbc
        myeloblast_percentages.append(pct)
        blast_percents.append(pct)
    else:
        blast_percents.append(0)

    wbc_counts.append(total_wbc)
    total_cell_counts.append(total_cells)
    slide_names.append(slide_name)

    print(f"\nImage: {slide_name}", file=indv_rep_file)
    if (b+n+er+eo+m+wo) > 0:
        print(f"  Myeloblast / total WBCs:                   {100*m/(b+n+er+eo+m+wo):.2f}%", file=indv_rep_file)
    else:
        print(f"  Myeloblast / total WBCs:                   N/A (no WBCs)", file=indv_rep_file)

#*****************************************************************************#
"""
Diagnostic Plots.
"""
if len(myeloblast_percentages) > 0:

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Histogram - Myeloblast Percentage Distribution
    bins = np.arange(0, 105, 5)
    axs[0].hist(myeloblast_percentages, bins=bins)
    axs[0].axvline(x=20, linestyle='--', linewidth=2)
    axs[0].set_title("Myeloblast Percentage Distribution")
    axs[0].set_xlabel("Myeloblast Percentage")
    axs[0].set_ylabel("Number of Slides")

    # (2) Bar chart - Total cell counts by type
    cell_labels = ["Myeloblast", "Basophil", "Neutrophil", "Eosinophil", "Erythroblast", "WBC Other", "RBC"]
    cell_values = [cell_counter[label] for label in cell_labels]

    axs[1].bar(cell_labels, cell_values)
    axs[1].set_title("Total Cell Counts by Type")
    axs[1].set_xlabel("Cell Type")
    axs[1].set_ylabel("Total Count")
    axs[1].tick_params(axis='x', rotation=30)
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("diagnostic_plots.png")
    plt.close()

#*****************************************************************************#
"""
Print myeloblast ratios and cell counters.
"""
print( ("% of  (# Myeloblasts)\n" +  
        "       -------------  = " + 
        str(100*cell_counter["Myeloblast"]/(cell_counter["Basophil"] + cell_counter["Neutrophil"] + cell_counter["Erythroblast"] + cell_counter["Eosinophil"] + cell_counter["Myeloblast"] + cell_counter["WBC Other"])) + "\n" +
        "       (total potential nucleated RBCs + WBCs)   \n"), 
      file=rep_file)

print( ("% of  (# Myeloblasts + # potential nucleated RBCs)\n" +  
        "       ------------------------------------------  = " + 
        str(100*(cell_counter["Myeloblast"]+cell_counter["Erythroblast"])/(cell_counter["Basophil"] + cell_counter["Neutrophil"] + cell_counter["Erythroblast"] + cell_counter["Eosinophil"] + cell_counter["Myeloblast"] + cell_counter["WBC Other"])) + "\n" +
        "        (total WBCs + # potential nucleated RBCs)   \n"), 
      file=rep_file)
print("\nMyeloblasts:           ", cell_counter["Myeloblast"], file=rep_file)
print("Other types of WBCs:   ", cell_counter["WBC Other"], file=rep_file)
print("Basophil:              ", cell_counter["Basophil"], file=rep_file)
print("Neutrophil:            ", cell_counter["Neutrophil"], file=rep_file)
print("Erythroblast:          ", cell_counter["Erythroblast"], file=rep_file)
print("Eosinophil:            ", cell_counter["Eosinophil"], file=rep_file)
print("RBCs:                  ", cell_counter["RBC"], file=rep_file)

#*****************************************************************************#
"""
Output total program runtime.
"""
program_end = time.perf_counter()
total_runtime = program_end - program_start
print(f"TOTAL PROGRAM TIME:    {total_runtime:.2f} seconds", file=rep_file)

# print runtime to terminal
print(f"\nTotal runtime: {total_runtime:.2f} seconds")

# yay :)
rep_file.close()
indv_rep_file.close()
