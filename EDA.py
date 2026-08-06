import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def target_predictor_separation(df, target):

    """
    Accepts a raw_data argument with the target variable and split into predictor dataframe "X" and target dataframe "y".
    Argument 1: Raw data dataframe
    Argument 2: Target dataframe
    """

    y = df[[target]]
    X = df.drop(columns=target)

    return X, y

def separate_num_cat(X):

    """
    Accepts a Predictor dataframe and separates it into two dataframes,
    First one will be all the numeric predictor columns,
    second one will be all the categorical predictor columns
    """

    X_num = X.select_dtypes(include=['int64', 'float64']).columns
    X_cat = X.select_dtypes(include=['category', 'object']).columns
    
    return X_num, X_cat

def plot_countplt(df, section, categorical, show_chart=False):
    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(8, 5))

    ax=sns.countplot(data=df, x=categorical, hue=categorical, palette='viridis')

    plt.title(f'{section}: {categorical} Distribution', fontsize=14, pad=15)
    plt.xlabel(f'{categorical} Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(f"plots/{section}_{categorical}_distribution.png", dpi=300, bbox_inches='tight')
    if show_chart:
        plt.show()
    plt.close()

def plot_histplot(df, numerical,show_chart=False):
    plt.figure(figsize=(9, 4))

    sns.histplot(data=df, x=numerical, kde=True, color='royalblue', bins=20)

    plt.axvline(df[numerical].mean(), color='crimson', linestyle='--', label=f'Mean: {df[numerical].mean(): .2f}')
    plt.axvline(df[numerical].median(), color='green', linestyle='-', label=f'Median: {df[numerical].median(): .2f}')

    plt.title(f'Numeric: Distribution of "{numerical}"', fontsize=14)
    plt.xlabel(numerical, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/Numerical_{numerical}_distribution.png", dpi=300, bbox_inches='tight')
    if show_chart:
        plt.show()
    plt.close()

def check_outlier(df, numerical, show_chart=False):
    fig, axes = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios': [1,3]})
    
    sns.boxplot(data=df, y=numerical, ax=axes[0], color='lightcoral')
    axes[0].set_title(f'Box Plot of {numerical}')

    sns.scatterplot(data=df, x=df.index, y=numerical, ax=axes[1], color='darkred', alpha=0.6)
    axes[1].set_title(f'Value Distribution of {numerical}')

    Q1 = df[numerical].quantile(0.25)
    Q3 = df[numerical].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    condition = (df[numerical] < lower_bound) | (df[numerical] > upper_bound)
    outliers = df[condition]
    print(f"Feature '{numerical}': Found {len(outliers)} outliers out of {len(df)} records ({len(outliers)/len(df)*100:.1f}%)")
    
    plt.tight_layout()
    plt.savefig(f"plots/Outlier_{numerical}_distribution.png", dpi=300, bbox_inches='tight')
    if show_chart:
        plt.show()
    plt.close()

def check_correlation(df, numerical, show_chart=False):
    corr_matrix = df[numerical].corr(method='pearson')
    plt.figure(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask, 
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink":.75}
    )

    plt.title('Numeric vs. Numeric Correlation Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"plots/Numeric vs. Numeric Correlation Heatmap.png", dpi=300, bbox_inches='tight')
    if show_chart:
        plt.show()    
    plt.close()


def chi_squared_test(contingency_table):
    """
    Perform chi_squared_test using raw absolute frequencies.
    """
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-squared Statistics: {chi2:.4f}")
    print(f"p-value: {p_val:.4e}") 

    if p_val < 0.05:
        print("Result: Statistically SIGNIFICANT association (p < 0.05)")
    else:
        print("Result: Statistically INSIGNIFICANT association (p >= 0.05)")
    print("=" * 20)

def plot_categorical_relationship(categorical, show_chart=False): 
    sns.set_theme(style='whitegrid')
    
    for i in range(len(categorical)):
        for j in range(i + 1, len(categorical)):
            col = categorical[i]
            target_col = categorical[j]
            
            print("\n" + "="*20)
            print(f"Statistical Association: {col} vs {target_col}")
            print("=" * 20)

            contingency_table = pd.pivot_table(
                df,
                index=col,
                columns=target_col,
                aggfunc='size',
                fill_value=0
            )

            contingency_pct = pd.crosstab(
                df[col],
                df[target_col], 
                normalize='index'
            ) * 100

            print("--- Normalized Distribution (% row-wise) ---")
            print(contingency_pct.round(2))
            print("=" * 20)

            chi_squared_test(contingency_table)

            plt.figure(figsize=(10, 5))
            ax = sns.countplot(data=df, x=col, hue=target_col, palette="Set2")

            plt.title(f"Bivariate Distribution of {col} segmented by {target_col}", fontsize=13, pad=15)
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Count', fontsize=11)
            plt.legend(title=target_col, loc='upper right')

            if df[col].nunique() > 4:
                plt.xticks(rotation=30, ha='right')

            plt.tight_layout()
            plt.savefig(f"plots/Bivariate Distribution of {col} segmented by {target_col}.png", dpi=300, bbox_inches='tight')

            if show_chart:
                plt.show()
            plt.close()

def numeric_vs_categorical(df, numerical, categorical, show_chart=False):
    """
    Systematically plots every numeric column segmented by every categorical column 
    in the dataframe to uncover all cross-type relationships.
    """

    sns.set_theme(style="whitegrid")
    
    for cat_col in categorical:
        print("\n" + "="*60)
        print(f" CATEGORICAL SEGMENT: {cat_col} vs. ALL NUMERIC FEATURES")
        print("="*60)
        
        for num_col in numerical:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            
            sns.boxplot(
                data=df, 
                x=cat_col, 
                y=num_col, 
                ax=axes[0], 
                palette="Set2",
                hue=cat_col,
                legend=False
            )
            axes[0].set_title(f'Box Plot: {num_col} by {cat_col}', fontsize=12)
            axes[0].set_xlabel(cat_col)
            axes[0].set_ylabel(num_col)
            
            sns.violinplot(
                data=df, 
                x=cat_col, 
                y=num_col, 
                ax=axes[1], 
                palette="Set2",
                hue=cat_col,
                legend=False
            )
            axes[1].set_title(f'Density Shape: {num_col} by {cat_col}', fontsize=12)
            axes[1].set_xlabel(cat_col)
            axes[1].set_ylabel(num_col)
            
            if df[cat_col].nunique() > 4:
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
            
            plt.suptitle(f'Cross-Type Analysis: {num_col} segmented by {cat_col}', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(f"plots/Num vs Cat_{num_col} segmented by {cat_col}.png", dpi=300, bbox_inches='tight')
            if show_chart:
                plt.show()
            plt.close()

def analyze_predictor_target_relationships(df, target, target_type, X_num, X_cat, show_chart=False):
    """
    Evaluates relationships between predictors and target based on target_type.
    
    Parameters:
    - df: pandas DataFrame containing the dataset
    - target: str, name of the target column
    - target_type: str, 'categorical' (classification) or 'numeric' (regression)
    - X_num: Index or list of numerical predictor column names
    - X_cat: Index or list of categorical predictor column names
    - show_chart: bool, whether to render charts in the interactive window
    """
    sns.set_theme(style="whitegrid")
    
    # Filter target out of predictor collections if present
    num_predictors = [col for col in X_num if col != target]
    cat_predictors = [col for col in X_cat if col != target]

    # =========================================================================
    # CASE 1: CATEGORICAL TARGET (Classification)
    # =========================================================================
    if target_type.lower() == 'categorical':
        print("\n" + "=" * 60)
        print(f" TARGET RELATIONSHIP ANALYSIS (Categorical Target: '{target}')")
        print("=" * 60)

        # 1A. Numeric Predictors vs Categorical Target
        print("\n--- [1/2] NUMERIC PREDICTORS vs TARGET ---")
        for num_col in num_predictors:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            
            # Box plot
            sns.boxplot(
                data=df, x=target, y=num_col, ax=axes[0], 
                palette="Set2", hue=target, legend=False
            )
            axes[0].set_title(f'Box Plot: {num_col} by {target}', fontsize=12)
            axes[0].set_xlabel(target)
            axes[0].set_ylabel(num_col)
            
            # Violin plot (Density Distribution)
            sns.violinplot(
                data=df, x=target, y=num_col, ax=axes[1], 
                palette="Set2", hue=target, legend=False
            )
            axes[1].set_title(f'Density Shape: {num_col} by {target}', fontsize=12)
            axes[1].set_xlabel(target)
            axes[1].set_ylabel(num_col)
            
            if df[target].nunique() > 4:
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')

            plt.suptitle(f'Predictor Strength: {num_col} vs Target ({target})', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(f"target_plots/Target_Categorical_vs_Num_{num_col}.png", dpi=300, bbox_inches='tight')
            
            if show_chart:
                plt.show()
            plt.close()

        # 1B. Categorical Predictors vs Categorical Target
        print("\n--- [2/2] CATEGORICAL PREDICTORS vs TARGET ---")
        for cat_col in cat_predictors:
            print("\n" + "=" * 20)
            print(f"Statistical Association: {cat_col} vs Target ({target})")
            print("=" * 20)
            
            # Contingency table for Chi-Square test
            contingency_table = pd.pivot_table(
                df, index=cat_col, columns=target, aggfunc='size', fill_value=0
            )
            
            # Normalized distribution for review
            contingency_pct = pd.crosstab(
                df[cat_col], df[target], normalize='index'
            ) * 100
            
            print("--- Normalized Distribution (% row-wise) ---")
            print(contingency_pct.round(2))
            print("=" * 20)
            
            # Call your existing chi_squared_test helper function
            chi_squared_test(contingency_table)
            
            # Visual count plot
            plt.figure(figsize=(10, 5))
            ax = sns.countplot(data=df, x=cat_col, hue=target, palette="Set2")
            plt.title(f"Bivariate Distribution of {cat_col} segmented by Target '{target}'", fontsize=13, pad=15)
            plt.xlabel(cat_col, fontsize=11)
            plt.ylabel('Count', fontsize=11)
            plt.legend(title=target, loc='upper right')
            
            if df[cat_col].nunique() > 4:
                plt.xticks(rotation=30, ha='right')
                
            plt.tight_layout()
            plt.savefig(f"target_plots/Target_Categorical_vs_Cat_{cat_col}.png", dpi=300, bbox_inches='tight')
            
            if show_chart:
                plt.show()
            plt.close()

    # =========================================================================
    # CASE 2: NUMERIC TARGET (Regression)
    # =========================================================================
    elif target_type.lower() == 'numeric':
        print("\n" + "=" * 60)
        print(f" TARGET RELATIONSHIP ANALYSIS (Numeric Target: '{target}')")
        print("=" * 60)

        # 2A. Numeric Predictors vs Numeric Target
        print("\n--- [1/2] NUMERIC PREDICTORS vs TARGET ---")
        all_num_cols = list(num_predictors) + [target]
        corr_series = df[all_num_cols].corr(method='pearson')[target].drop(target)
        
        # Pearson Correlation Overview Bar Plot
        plt.figure(figsize=(8, max(4, len(num_predictors) * 0.4)))
        corr_series.sort_values().plot(kind='barh', color='skyblue')
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.title(f"Pearson Correlation Strength vs Target '{target}'", fontsize=14, pad=15)
        plt.xlabel("Pearson Correlation Coefficient (-1 to +1)", fontsize=12)
        plt.ylabel("Numeric Predictors", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"target_plots/Target_Numeric_vs_Num_Correlations.png", dpi=300, bbox_inches='tight')
        
        if show_chart:
            plt.show()
        plt.close()

        # Scatter plot for each individual numeric predictor vs target
        for num_col in num_predictors:
            plt.figure(figsize=(8, 4.5))
            sns.scatterplot(data=df, x=num_col, y=target, alpha=0.6, color='teal')
            sns.regplot(data=df, x=num_col, y=target, scatter=False, color='crimson')
            plt.title(f"Scatter Plot: '{num_col}' vs Target '{target}'", fontsize=14, pad=15)
            plt.xlabel(num_col, fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.tight_layout()
            plt.savefig(f"target_plots/Target_Numeric_vs_Num_Scatter_{num_col}.png", dpi=300, bbox_inches='tight')
            
            if show_chart:
                plt.show()
            plt.close()

        # 2B. Categorical Predictors vs Numeric Target
        print("\n--- [2/2] CATEGORICAL PREDICTORS vs TARGET ---")
        for cat_col in cat_predictors:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            
            # Box plot
            sns.boxplot(
                data=df, x=cat_col, y=target, ax=axes[0], 
                palette="Set2", hue=cat_col, legend=False
            )
            axes[0].set_title(f'Box Plot: Target ({target}) by {cat_col}', fontsize=12)
            axes[0].set_xlabel(cat_col)
            axes[0].set_ylabel(target)
            
            # Violin plot
            sns.violinplot(
                data=df, x=cat_col, y=target, ax=axes[1], 
                palette="Set2", hue=cat_col, legend=False
            )
            axes[1].set_title(f'Density Shape: Target ({target}) by {cat_col}', fontsize=12)
            axes[1].set_xlabel(cat_col)
            axes[1].set_ylabel(target)
            
            if df[cat_col].nunique() > 4:
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')

            plt.suptitle(f"Target Analysis: Target '{target}' by Category '{cat_col}'", fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(f"plots/Target_Numeric_vs_Cat_{cat_col}.png", dpi=300, bbox_inches='tight')
            
            if show_chart:
                plt.show()
            plt.close()

    else:
        raise ValueError("Invalid target_type. Please pass either 'categorical' or 'numeric'.")
    
def data_overview(df, target, target_type, print_X_y = True, show_chart=False):

    """
    Accepts a dataframe, seperates into predictor and target columns, and return numeric and categorical columns list. 
    This function also describes the dimension of the predictor dataframe
    """

    X, y = target_predictor_separation(df, target)
    X_num, X_cat = separate_num_cat(X)

    print(f"There are {len(X.columns)} predictor columns")
    print(f"{len(X_num)} of them are numerical.")
    print(f"{len(X_cat)} of them are categorical.\n")

    print("="*20, "df summary: ", "="*20)
    X.info()
    

    if print_X_y:   
        print("\n", "="*20, "Predictors", "="*20)
        print(X.head())

        print("\n", "="*20, "Target", "="*20)
        print(y.head())

    print("---Raw Target Counts---")
    print(df[target].value_counts())

    print("\n---Target Percentages---")
    print(df[target].value_counts(normalize=True) * 100)

    #Review the distribution of target
    plot_countplt(y, "Target_", target, show_chart=show_chart)
    
    #Review numerical columns' distributions:
    for numerical in X_num:
        plot_histplot(X, numerical,show_chart=show_chart)
        check_outlier(X, numerical,show_chart=show_chart)

    print(X.describe().T)

    for categorical in X_cat:
        print(f"\n--- Frequency Table for {categorical} ---")

        freq_df = pd.DataFrame(
            {
                'Count': df[categorical].value_counts(),
                'Percentage (%)': df[categorical].value_counts(normalize=True) * 100
            }
        )
        print(freq_df)     
        plot_countplt(df, "Categorical_", categorical, show_chart=show_chart)  
    
    check_correlation(df,X_num,)
    plot_categorical_relationship(X_cat, show_chart=show_chart)
    numeric_vs_categorical(df, X_num, X_cat, show_chart=show_chart)
    analyze_predictor_target_relationships(
            df=df, 
            target=target, 
            target_type=target_type, 
            X_num=X_num, 
            X_cat=X_cat, 
            show_chart=show_chart
        )
    return X, y, X_num, X_cat

def count_missing_data(df):
    print("=" * 55)
    print(" 1. MISSING DATA PER COLUMN")
    print("=" * 55)

    col_missing_count = df.isnull().sum()
    col_missing_pct = (df.isnull().sum() / len(df)) * 100

    missing_col_df = pd.DataFrame({
        'Missing Count': col_missing_count,
        'Percentage (%)': col_missing_pct
    }).sort_values(by='Percentage (%)', ascending=False)

    # Display columns that have missing values
    print(missing_col_df[missing_col_df['Missing Count'] > 0])
    if (col_missing_count == 0).all():
        print("No missing values found in any column.")

    # =========================================================
    # 2. TOTAL PERCENTAGE OF ROWS CONTAINING MISSING DATA
    # =========================================================
    print("\n" + "=" * 55)
    print(" 2. ROW-LEVEL MISSINGNESS OVERVIEW")
    print("=" * 55)

    # df.isnull().any(axis=1) creates a boolean mask where True = row has at least 1 missing cell
    rows_with_missing = df.isnull().any(axis=1).sum()
    total_rows = len(df)
    row_missing_pct = (rows_with_missing / total_rows) * 100

    complete_rows = total_rows - rows_with_missing
    complete_rows_pct = 100 - row_missing_pct

    print(f"Total Rows in Dataset            : {total_rows}")
    print(f"Complete Rows (No Missing Values): {complete_rows} ({complete_rows_pct:.2f}%)")
    print(f"Rows with >= 1 Missing Value     : {rows_with_missing} ({row_missing_pct:.2f}%)")
    print("=" * 55)
    
if __name__ == "__main__":

    df = pd.read_csv("train.csv", index_col='id')
    count_missing_data(df)
    X, y, X_num, X_cat = data_overview(df, target='addicted_label', target_type='categorical', print_X_y= False, show_chart=False)
