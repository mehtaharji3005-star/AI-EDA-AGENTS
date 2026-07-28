
import pandas as pd


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    """
    print("=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)

    # 1. Basic Dataset Shape
    print("\n[1] DATASET SHAPE")
    print(f"Number of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    # 2. Column Names and Data Types
    print("\n[2] COLUMNS AND DATA TYPES")
    dtype_df = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
        }
    )
    print(dtype_df.to_string(index=False))

    # 3. Missing Values Analysis
    print("\n[3] MISSING VALUES ANALYSIS")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)

    missing_df = pd.DataFrame(
        {"Missing Values": missing_count, "Percentage (%)": missing_percent}
    )
    # Filter to show only columns with missing values, or show all if none
    if missing_count.sum() > 0:
        print(missing_df[missing_df["Missing Values"] > 0].to_string())
    else:
        print("🎉 Great news! There are no missing values in this dataset.")

    # 4. Duplicate Rows
    print("\n[4] DUPLICATE ROWS")
    duplicate_count = df.duplicated().sum()
    duplicate_percent = (duplicate_count / len(df)) * 100
    print(
        f"Number of duplicate rows: {duplicate_count} ({duplicate_percent:.2f}% of total rows)"
    )

    # 5. Statistical Summary for Numerical Columns
    print("\n[5] STATISTICAL SUMMARY (Numerical Columns)")
    num_cols = df.select_dtypes(include=["number"])
    if not num_cols.empty:
        print(num_cols.describe().T.to_string())
    else:
        print("No numerical columns found in the dataset.")

    # 6. Statistical Summary for Categorical Columns
    print("\n[6] STATISTICAL SUMMARY (Categorical Columns)")
    cat_cols = df.select_dtypes(include=["object", "category"])
    if not cat_cols.empty:
        print(cat_cols.describe().T.to_string())
    else:
        print("No categorical columns found in the dataset.")

    print("\n" + "=" * 60)
    print(" END OF EDA REPORT")
    print("=" * 60)
