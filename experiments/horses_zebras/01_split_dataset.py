from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(data_dir, classes, random_seed=42, val_size=0.1, test_size=0.1):
    """
    Turn a dataset from a single folder into a train, validation, and test set.
    """
    data_dir = Path(data_dir)

    for folder in ["train", "val", "test"]:
        (data_dir / folder).mkdir(parents=True, exist_ok=True)

    for class_name in classes:
        class_dir = data_dir / class_name
        image_files = list(class_dir.glob("*.jpg"))
        train_files, test_files = train_test_split(
            image_files, test_size=test_size, random_state=random_seed
        )
        train_files, val_files = train_test_split(
            train_files, test_size=val_size / (1 - test_size), random_state=random_seed
        )

        for folder, files in zip(
            ["train", "val", "test"], [train_files, val_files, test_files]
        ):
            new_class_dir = data_dir / folder / class_name
            print(f"Moving {len(files)} files to {new_class_dir}")
            new_class_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                file.rename(new_class_dir / file.name)
        # Remove empty class folder
        class_dir.rmdir()


if __name__ == "__main__":
    split_dataset(
        "/nrs/funke/adjavond/data/duplex/horses_zebras", classes=["horses", "zebras"]
    )
