
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Extract the file extension and convert to lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Map extensions to their respective pandas read functions
    try:
        if file_extension == ".csv":
            # You can add extra parameters like encoding if needed
            return pd.read_csv(file_path)

        elif file_extension in [".xls", ".xlsx"]:
            return pd.read_excel(file_path)

        elif file_extension == ".json":
            return pd.read_json(file_path)

        elif file_extension in [".parquet", ".pq"]:
            return pd.read_parquet(file_path)

        elif file_extension == ".tsv":
            return pd.read_csv(file_path, sep="\t")

        elif file_extension in [".h5", ".hdf5"]:
            return pd.read_hdf(file_path)

        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


# --- Example Usage ---
# df = read_uploaded_file('path/to/your/uploaded_file.csv')
# print(df.head())
