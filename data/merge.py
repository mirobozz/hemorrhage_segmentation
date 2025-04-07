import os
import shutil
import my_utils.config as cfg

def merge_datasets(source1, source2, output_dir):
    """
    Merge two dataset directories containing 'images' and 'masks' subdirectories into a single output directory.
    Handles duplicate filenames by appending an index.

    :param source1: Path to the first dataset directory (e.g., 'train/')
    :param source2: Path to the second dataset directory (e.g., 'valid/')
    :param output_dir: Path to the output dataset directory
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    def copy_files(src_dir, category):
        """
        Copy images or masks from a source directory into the output directory.
        Handles duplicate filenames.
        """
        dest_dir = os.path.join(output_dir, category)
        for filename in os.listdir(os.path.join(src_dir, category)):
            src_path = os.path.join(src_dir, category, filename)
            if os.path.isfile(src_path):
                base, ext = os.path.splitext(filename)
                dst_path = os.path.join(dest_dir, filename)

                # Ensure unique filenames if duplicates exist
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")

    # Copy images and masks from both datasets
    for dataset in [source1, source2]:
        copy_files(dataset, "images")
        copy_files(dataset, "masks")

    print(f"Dataset merge complete. Merged data saved in {output_dir}")


# Example usage
merge_datasets('D:/bp_dataset/base_subsets/valid/',

               'D:/bp_dataset/base_subsets/test/',

               'D:/bp_dataset/combined_subsets/valid_test')
