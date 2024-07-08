import midv500
import os
import json 

def download_data(dataset_dir):
    flag=False
    try:
        #Downloading the dataset in the output directory
        midv500.download_dataset(dataset_dir)
        flag=True
        print("/n Dataset Downloaded successfully")
    except:
        pass 

    return flag


# Helper function to read JSON files
def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['quad']  # Adjust according to your JSON structure


def load_data(base_dir):
    image_paths = []
    corner_coords = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        images_dir = os.path.join(folder_path, "images")
        ground_truth_dir = os.path.join(folder_path, "ground_truth")

        for doc_type in os.listdir(images_dir):
            images_path = os.path.join(images_dir, doc_type)
            ground_truth_path = os.path.join(ground_truth_dir, doc_type)

            if os.path.isdir(images_path) and os.path.isdir(ground_truth_path):
                for img_file in os.listdir(images_path):
                    if img_file.endswith(".tif"):
                        img_path = os.path.join(images_path, img_file)
                        json_file = img_file.replace(".tif", ".json")
                        json_path = os.path.join(ground_truth_path, json_file)

                        if os.path.exists(json_path):
                            coords = read_json(json_path)
                            # Flatten the list of coordinates
                            coords = [item for sublist in coords for item in sublist]
                            image_paths.append(img_path)
                            corner_coords.append(coords)
    
    return image_paths, corner_coords