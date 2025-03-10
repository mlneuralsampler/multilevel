import os
import zipfile


def file_exists_or_unzip(file_path, zip_folder):
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"✅ File '{file_path}' found. Continuing...")
        return

    print(f"⚠️ File '{file_path}' not found. Unzipping files in '{zip_folder}'...")

    # Unzip all .zip files in the folder
    for zip_file in os.listdir(zip_folder):
        if zip_file.endswith(".zip"):
            zip_path = os.path.join(zip_folder, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_folder)
            print(f"✅ Extracted '{zip_file}'")

    # Check again if the file exists after extraction
    if os.path.exists(file_path):
        print(f"✅ File '{file_path}' found after extraction. Continuing...")
    else:
        print(f"❌ File '{file_path}' still not found after extraction. Please check manually.")
