import os
import shutil
import random


# ==========================================
# 1. Dataset paths
# ==========================================

SOURCE_DIR = "dataset/Cotton-Dataset"

OUTPUT_DIR = "dataset_split"


# ==========================================
# 2. Split percentages
# ==========================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================
# 3. Random seed
# ==========================================

random.seed(42)


# ==========================================
# 4. Create folder function
# ==========================================

def create_folder(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# ==========================================
# 5. Split dataset
# ==========================================

def split_images():

    # Check whether dataset exists

    if not os.path.exists(SOURCE_DIR):

        print("Dataset folder not found!")

        print(
            "Expected location:",
            SOURCE_DIR
        )

        return


    # Get Stage-1, Stage-2, etc.

    stages = [

        folder

        for folder in os.listdir(SOURCE_DIR)

        if os.path.isdir(
            os.path.join(
                SOURCE_DIR,
                folder
            )
        )

    ]


    print("\nClasses found:")

    print(stages)


    # ======================================
    # Process each stage
    # ======================================

    for stage in stages:

        stage_path = os.path.join(
            SOURCE_DIR,
            stage
        )


        # Get image files

        images = [

            file

            for file in os.listdir(
                stage_path
            )

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )

        ]


        # Shuffle images

        random.shuffle(images)


        # Total images

        total = len(images)


        # Calculate split positions

        train_end = int(
            total * TRAIN_RATIO
        )


        validation_end = (
            train_end
            +
            int(
                total * VALIDATION_RATIO
            )
        )


        # Split images

        train_images = images[
            :train_end
        ]


        validation_images = images[
            train_end:validation_end
        ]


        test_images = images[
            validation_end:
        ]


        # ==================================
        # Display information
        # ==================================

        print("\n----------------------------")

        print("Stage:", stage)

        print("Total:", total)

        print(
            "Train:",
            len(train_images)
        )

        print(
            "Validation:",
            len(validation_images)
        )

        print(
            "Test:",
            len(test_images)
        )


        # ==================================
        # Destination folders
        # ==================================

        train_dir = os.path.join(

            OUTPUT_DIR,

            "train",

            stage

        )


        validation_dir = os.path.join(

            OUTPUT_DIR,

            "validation",

            stage

        )


        test_dir = os.path.join(

            OUTPUT_DIR,

            "test",

            stage

        )


        # Create folders

        create_folder(
            train_dir
        )

        create_folder(
            validation_dir
        )

        create_folder(
            test_dir
        )


        # ==================================
        # Copy training images
        # ==================================

        for image in train_images:

            source = os.path.join(

                stage_path,

                image

            )


            destination = os.path.join(

                train_dir,

                image

            )


            shutil.copy2(

                source,

                destination

            )


        # ==================================
        # Copy validation images
        # ==================================

        for image in validation_images:

            source = os.path.join(

                stage_path,

                image

            )


            destination = os.path.join(

                validation_dir,

                image

            )


            shutil.copy2(

                source,

                destination

            )


        # ==================================
        # Copy test images
        # ==================================

        for image in test_images:

            source = os.path.join(

                stage_path,

                image

            )


            destination = os.path.join(

                test_dir,

                image

            )


            shutil.copy2(

                source,

                destination

            )


    print(
        "\n================================"
    )

    print(
        "Dataset splitting completed!"
    )

    print(
        "================================"
    )


# ==========================================
# 6. Run program
# ==========================================

if __name__ == "__main__":

    split_images()