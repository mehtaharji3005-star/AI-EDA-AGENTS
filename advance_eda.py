
# advance_eda.py

def eda_by_ai(df):
    """
    Performs an exhaustive, end-to-end Advanced Exploratory Data Analysis (EDA) 
    on the provided pandas DataFrame `df`.
    """
    # Import all necessary libraries inside the function
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    
    # Suppress warnings for clean execution
    warnings.filterwarnings('ignore')
    
    # Set global plotting aesthetics
    sns.set_theme(style="whitegrid")
    
    print("=" * 60)
    print("PHASE 1: Setup and Data Overview")
    print("=" * 60)
    
    # 1. Basic dataset information
    print("\n--- DataFrame Info ---")
    print(df.info())
    
    # Missing values count and percentage per column
    missing_count = df.isnull().sum()
    missing_percentage = (df.isnull().mean()) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage (%)': missing_percentage})
    print("\n--- Missing Values ---")
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Duplicate rows count
    duplicate_count = df.duplicated().sum()
    print(f"\n--- Duplicate Rows Count: {duplicate_count} ---")
    
    # 2. Comprehensive descriptive statistics
    print("\n--- Descriptive Statistics (Numerical) ---")
    print(df.describe())
    
    cat_cols_init = df.select_dtypes(include=['O', 'category']).columns
    if len(cat_cols_init) > 0:
        print("\n--- Descriptive Statistics (Categorical) ---")
        print(df.describe(include=['O', 'category']))
    else:
        print("\n--- No Categorical Columns Found for Descriptive Statistics ---")

    print("\n" + "=" * 60)
    print("PHASE 2: Univariate Analysis")
    print("=" * 60)
    
    # 1. Numerical Columns Univariate Analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nDetected Numerical Columns: {numerical_cols}")
    
    for col in numerical_cols:
        # Handle missing values gracefully before plotting
        temp_data = df[col].dropna()
        if temp_data.empty:
            print(f"Skipping {col} due to all missing values.")
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram with KDE
        sns.histplot(temp_data, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Histogram & KDE of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        
        # Boxplot
        sns.boxplot(x=temp_data, ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot of {col}')
        axes[1].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()

    # 2. Object / Categorical Columns Univariate Analysis
    # Exclude high-cardinality ID columns (heuristic: unique values > 50% of length or ends with 'id')
    categorical_cols = [
        c for c in df.select_dtypes(include=['O', 'category']).columns 
        if df[c].nunique() < len(df) * 0.5 and not c.lower().endswith('id')
    ]
    print(f"\nDetected Categorical Columns (Filtered): {categorical_cols}")
    
    for col in categorical_cols:
        print(f"\nValue Counts for {col}:")
        print(df[col].value_counts().head(10))
        
        plt.figure(figsize=(8, 4))
        top_cats = df[col].value_counts().head(10).index
        sns.countplot(data=df[df[col].isin(top_cats)], x=col, order=top_cats, palette='viridis')
        plt.title(f'Count Plot of {col} (Top Categories)')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 60)
    print("PHASE 3: Correlation Analysis")
    print("=" * 60)
    
    if len(numerical_cols) > 1:
        # Compute Pearson correlation matrix
        corr_matrix = df[numerical_cols].corr(method='pearson')
        print("\nPearson Correlation Matrix:")
        print(corr_matrix)
        
        # Generate Correlation Heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap (Pearson)')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNot enough numerical columns to compute a correlation matrix.")

    print("\n" + "=" * 60)
    print("PHASE 4: Bivariate Analysis & Categorical Bar Plots with Hue")
    print("=" * 60)
    
    # Identify typical columns if available, else pick dynamically
    # Looking for columns matching standard business datasets (sales, region, segment, etc.)
    all_cols_lower = {c.lower(): c for c in df.columns}
    
    metric_col = all_cols_lower.get('sales', numerical_cols[0] if numerical_cols else None)
    cat_dim1 = all_cols_lower.get('region', categorical_cols[0] if len(categorical_cols) > 0 else None)
    cat_dim2 = all_cols_lower.get('segment', categorical_cols[1] if len(categorical_cols) > 1 else (categorical_cols[0] if len(categorical_cols) > 0 else None))
    
    print(f"Selected Metric for Bivariate Analysis: {metric_col}")
    print(f"Selected Primary Categorical Dimension: {cat_dim1}")
    print(f"Selected Secondary Categorical Dimension (Hue): {cat_dim2}")
    
    if metric_col and cat_dim1 and cat_dim2 and cat_dim1 != cat_dim2:
        plt.figure(figsize=(10, 6))
        # Grouped bar plot using seaborn barplot with hue
        sns.barplot(data=df, x=cat_dim1, y=metric_col, hue=cat_dim2, estimator=np.mean, ci=None, palette='muted')
        plt.title(f'Mean {metric_col} by {cat_dim1} and {cat_dim2}')
        plt.xlabel(cat_dim1)
        plt.ylabel(f'Mean {metric_col}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=cat_dim2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping grouped bar plot due to insufficient matching categorical/numerical columns.")
        
    # Scatter plots with regression lines between pairs of numerical variables
    if len(numerical_cols) >= 2:
        num_pair_1, num_pair_2 = numerical_cols[0], numerical_cols[1]
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df, x=num_pair_1, y=num_pair_2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'Regression Plot: {num_pair_1} vs {num_pair_2}')
        plt.xlabel(num_pair_1)
        plt.ylabel(num_pair_2)
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 60)
    print("PHASE 5: Time Series Analysis (Conditional)")
    print("=" * 60)
    
    # Check if any column contains date/datetime data or can be parsed
    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        else:
            # Try parsing a small sample to see if it's a date string
            try:
                sample = df[col].dropna().head(5)
                if not sample.empty and len(str(sample.iloc[0])) >= 8:
                    pd.to_datetime(sample, errors='raise')
                    date_col = col
                    break
            except (ValueError, TypeError):
                continue
                
    if date_col:
        print(f"Date column found: '{date_col}'. Proceeding with Time Series Analysis.")
        # Ensure it is datetime
        temp_ts_df = df.copy()
        temp_ts_df[date_col] = pd.to_datetime(temp_ts_df[date_col], errors='coerce')
        temp_ts_df = temp_ts_df.dropna(subset=[date_col])
        
        # Use numerical metric if available
        ts_metric = metric_col if metric_col else (numerical_cols[0] if numerical_cols else None)
        
        if ts_metric:
            # Resample / Aggregate over time (Monthly trend)
            ts_grouped = temp_ts_df.set_index(date_col).resample('M')[ts_metric].sum().reset_index()
            
            plt.figure(figsize=(12, 5))
            sns.lineplot(data=ts_grouped, x=date_col, y=ts_metric, marker='o', color='b')
            plt.title(f'Time Series Trend of {ts_metric} Over Time (Monthly)')
            plt.xlabel('Date')
            plt.ylabel(f'Total {ts_metric}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No numerical metric found for time series aggregation.")
    else:
        print("No date column found, skipping Time Series Analysis.")

    print("\n" + "=" * 60)
    print("PHASE 6: Multivariate Analysis")
    print("=" * 60)
    
    # Numerical vs Numerical vs Categorical (Scatter plot with hue)
    if len(numerical_cols) >= 2 and len(categorical_cols) >= 1:
        x_var, y_var = numerical_cols[0], numerical_cols[1]
        hue_var = categorical_cols[0]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue_var, alpha=0.7, palette='deep')
        plt.title(f'Multivariate Scatter: {x_var} vs {y_var} colored by {hue_var}')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    # Multi-variable grouping and tabular summary
    if len(categorical_cols) >= 2 and numerical_cols:
        grp_col1, grp_col2 = categorical_cols[0], categorical_cols[1]
        agg_metric = numerical_cols[0]
        
        print(f"\nMulti-variable Grouping Summary ({grp_col1}, {grp_col2}) on {agg_metric}:")
        grouped_summary = df.groupby([grp_col1, grp_col2])[agg_metric].agg(['mean', 'sum', 'count']).reset_index()
        print(grouped_summary.head(15))
        
        # Pivot table for heatmap representation
        try:
            pivot_table = grouped_summary.pivot(index=grp_col1, columns=grp_col2, values='mean')
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
            plt.title(f'Heatmap of Mean {agg_metric} by {grp_col1} and {grp_col2}')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not generate pivot heatmap: {e}")
    else:
        print("Skipping multi-variable grouping heatmap due to lack of distinct categorical dimensions.")

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)
