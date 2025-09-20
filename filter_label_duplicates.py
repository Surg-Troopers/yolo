import os

def remove_duplicates(folder1, folder2):
    """
    Remove duplicate files from folder2 if they also exist in folder1.
    """
    # Get list of files in each folder
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find duplicates
    duplicates = files1.intersection(files2)

    print(f"Found {len(duplicates)} duplicate(s).")

    for filename in duplicates:
        file_path = os.path.join(folder2, filename)
        try:
            os.remove(file_path)
            print(f"Removed {file_path}")
        except Exception as e:
            print(f"Could not remove {file_path}: {e}")

if __name__ == "__main__":
    folder1 = "Training_Pseudo/extra_img_set1_12347/images"   # <-- change this
    folder2 = "extra_img_91012/images"   # <-- change this
    
    remove_duplicates(folder1, folder2)
