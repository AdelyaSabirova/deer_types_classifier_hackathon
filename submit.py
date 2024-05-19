import os
import pandas as pd
from PIL import Image, ImageFile
from classifier import CLASS_p
import uuid
from datetime import datetime
from tqdm import tqdm

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the class mapping
class_mapping = {
    "Кабарга": 0,
    "Косуля": 1,
    "Олень": 2
}

def load_images_from_folder(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('jpeg', 'jpg', 'png')):
                image_files.append(os.path.join(root, file))
    return image_files

def save_image_to_folder(image, category, original_filename, base_folder):
    folder_path = os.path.join(base_folder, category)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = original_filename


    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(os.path.join(folder_path, unique_filename))

def classify_and_save_predictions(input_folder='input_images', output_folder=None):
    if output_folder is None:
        output_folder = datetime.now().strftime("images_%Y%m%d_%H%M%S")

    os.makedirs(output_folder, exist_ok=True)
    classifier = CLASS_p()
    all_files = load_images_from_folder(input_folder)
    class_counts = {0: 0, 1: 0, 2: 0}
    predictions = []

    for i, file in enumerate(tqdm(all_files)):
        try:
            with open(file, "rb") as f:
                image = Image.open(f).convert('RGB')
            original_filename = os.path.basename(file)
            category, _ = classifier.predict(image)
            category_label = class_mapping[category]
            save_image_to_folder(image, category, original_filename, output_folder)
            class_counts[category_label] += 1
            predictions.append((original_filename, category_label))
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, columns=['img_name', 'class'])
    csv_filename = os.path.join(output_folder, "predictions.csv")
    predictions_df.to_csv(csv_filename, index=False)

    print(f"Classification complete. Predictions saved to {csv_filename}.")
    print("Class counts:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify images and save predictions to CSV.")
    parser.add_argument('--input', type=str, default='input_images', help="Input folder with images.")
    parser.add_argument('--output', type=str, help="Output folder to save classified images and predictions CSV.")

    args = parser.parse_args()

    classify_and_save_predictions(input_folder="test_minprirodi_Parnokopitnie", output_folder=args.output)


