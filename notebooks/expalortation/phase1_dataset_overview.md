# 📊 Phase 1: Data Overview - Electronics Sales Analysis

## 🧩 Tasks Checklist:
- [ ] **Step 1**: Load the data - Ensure it reads cleanly without encoding issues
- [ ] **Step 2**: Inspect shape - Know how many rows and columns 
- [ ] **Step 3**: Check column names & data types - Identify categorical vs numerical features
- [ ] **Step 4**: Preview first rows - Verify content and structure
- [ ] **Step 5**: Check for missing/null values - Detect incomplete records early
- [ ] **Step 6**: Check for duplicates - Remove repeated rows if any
- [ ] **Step 7**: Validate date parsing - Ensure "Purchase Date" is recognized as datetime

---
**Dataset**: Electronic Sales (Sep 2023 - Sep 2024)  
**Expected Records**: ~20,000 transactions


```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("✅ Libraries imported successfully!")
print(f"📅 Analysis started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

    ✅ Libraries imported successfully!
    📅 Analysis started on: 2025-10-08 23:48:14
    


```python
# Setup figure saving configuration
import os
from pathlib import Path

# Create figure directories if they don't exist
FIGURE_DIR = Path("../outputs/figures")
FIGURE_SUBDIRS = {
    'exploratory': FIGURE_DIR / 'exploratory',
    'correlations': FIGURE_DIR / 'correlations', 
    'distributions': FIGURE_DIR / 'distributions',
    'model_performance': FIGURE_DIR / 'model_performance',
    'business_insights': FIGURE_DIR / 'business_insights'
}

# Create all directories
for subdir_name, subdir_path in FIGURE_SUBDIRS.items():
    subdir_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created: {subdir_path}")

# Configure matplotlib for high-quality figure saving
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'

# Helper function to save figures
def save_figure(fig, filename, category='exploratory', formats=['png', 'pdf']):
    """Save figure to appropriate directory with multiple formats"""
    save_dir = FIGURE_SUBDIRS[category]
    
    for fmt in formats:
        filepath = save_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {filepath}")

print("\n✅ Figure saving configuration complete!")
print(f"📊 Main figures directory: {FIGURE_DIR.absolute()}")
print(f"📈 Available categories: {list(FIGURE_SUBDIRS.keys())}")
```

    📁 Created: ..\outputs\figures\exploratory
    📁 Created: ..\outputs\figures\correlations
    📁 Created: ..\outputs\figures\distributions
    📁 Created: ..\outputs\figures\model_performance
    📁 Created: ..\outputs\figures\business_insights
    
    ✅ Figure saving configuration complete!
    📊 Main figures directory: c:\Users\hatim\OneDrive\سطح المكتب\iau\25.26\Data_Mining\Project\DataMining\notebooks\..\outputs\figures
    📈 Available categories: ['exploratory', 'correlations', 'distributions', 'model_performance', 'business_insights']
    

## 📁 Step 1: Load the Data
**Goal**: Ensure the CSV file reads cleanly without encoding issues


```python
# Step 1: Load the data with error handling
try:
    # Define file path
    data_path = "../data/raw/Electronic_sales_Sep2023-Sep2024.csv"
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    print("✅ Step 1 COMPLETED: Data loaded successfully!")
    print(f"📄 File path: {data_path}")
    print(f"📊 Initial data loaded: {len(df)} records")
    
except FileNotFoundError:
    print("❌ Error: CSV file not found. Check the file path.")
except UnicodeDecodeError:
    print("⚠️  Encoding issue detected. Trying with different encoding...")
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
        print("✅ Data loaded with latin-1 encoding")
    except:
        print("❌ Failed to load with alternative encoding")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

    ✅ Step 1 COMPLETED: Data loaded successfully!
    📄 File path: ../data/raw/Electronic_sales_Sep2023-Sep2024.csv
    📊 Initial data loaded: 20000 records
    

## 📐 Step 2: Inspect Shape
**Goal**: Know how many rows and columns we have


```python
# Step 2: Inspect dataset shape
print("✅ Step 2 COMPLETED: Dataset Shape Analysis")
print("=" * 50)
print(f"📊 Dataset Shape: {df.shape}")
print(f"📈 Total Records (Rows): {df.shape[0]:,}")
print(f"📋 Total Features (Columns): {df.shape[1]}")
print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Quick size validation
expected_records = 20000
actual_records = df.shape[0]
if actual_records == expected_records:
    print(f"✅ Record count matches expectation: {expected_records:,}")
else:
    print(f"⚠️  Record count differs from expected:")
    print(f"   Expected: {expected_records:,}")
    print(f"   Actual: {actual_records:,}")
    print(f"   Difference: {abs(actual_records - expected_records):,}")
```

    ✅ Step 2 COMPLETED: Dataset Shape Analysis
    ==================================================
    📊 Dataset Shape: (20000, 16)
    📈 Total Records (Rows): 20,000
    📋 Total Features (Columns): 16
    💾 Memory Usage: 10.90 MB
    ✅ Record count matches expectation: 20,000
    

## 🏷️ Step 3: Column Names & Data Types
**Goal**: Identify categorical vs numerical features


```python
# Step 3: Analyze column names and data types
print("✅ Step 3 COMPLETED: Column Names & Data Types Analysis")
print("=" * 60)

# Display basic info
print("📋 DATASET INFO:")
df.info(memory_usage='deep')

print("\n" + "=" * 60)
print("🏷️  COLUMN ANALYSIS:")

# Categorize columns by data type
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

print(f"\n📊 NUMERIC COLUMNS ({len(numeric_cols)}):")
for i, col in enumerate(numeric_cols, 1):
    print(f"   {i}. {col} ({df[col].dtype})")

print(f"\n🏷️  CATEGORICAL/TEXT COLUMNS ({len(categorical_cols)}):")
for i, col in enumerate(categorical_cols, 1):
    unique_count = df[col].nunique()
    print(f"   {i}. {col} ({df[col].dtype}) - {unique_count} unique values")

if datetime_cols:
    print(f"\n📅 DATETIME COLUMNS ({len(datetime_cols)}):")
    for i, col in enumerate(datetime_cols, 1):
        print(f"   {i}. {col} ({df[col].dtype})")
else:
    print(f"\n📅 DATETIME COLUMNS (0): No datetime columns detected yet")
```

    ✅ Step 3 COMPLETED: Column Names & Data Types Analysis
    ============================================================
    📋 DATASET INFO:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20000 entries, 0 to 19999
    Data columns (total 16 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Customer ID        20000 non-null  int64  
     1   Age                20000 non-null  int64  
     2   Gender             19999 non-null  object 
     3   Loyalty Member     20000 non-null  object 
     4   Product Type       20000 non-null  object 
     5   SKU                20000 non-null  object 
     6   Rating             20000 non-null  int64  
     7   Order Status       20000 non-null  object 
     8   Payment Method     20000 non-null  object 
     9   Total Price        20000 non-null  float64
     10  Unit Price         20000 non-null  float64
     11  Quantity           20000 non-null  int64  
     12  Purchase Date      20000 non-null  object 
     13  Shipping Type      20000 non-null  object 
     14  Add-ons Purchased  15132 non-null  object 
     15  Add-on Total       20000 non-null  float64
    dtypes: float64(3), int64(4), object(9)
    memory usage: 10.9 MB
    
    ============================================================
    🏷️  COLUMN ANALYSIS:
    
    📊 NUMERIC COLUMNS (7):
       1. Customer ID (int64)
       2. Age (int64)
       3. Rating (int64)
       4. Total Price (float64)
       5. Unit Price (float64)
       6. Quantity (int64)
       7. Add-on Total (float64)
    
    🏷️  CATEGORICAL/TEXT COLUMNS (9):
       1. Gender (object) - 2 unique values
       2. Loyalty Member (object) - 2 unique values
       3. Product Type (object) - 5 unique values
       4. SKU (object) - 10 unique values
       5. Order Status (object) - 2 unique values
       6. Payment Method (object) - 6 unique values
       7. Purchase Date (object) - 366 unique values
       8. Shipping Type (object) - 5 unique values
       9. Add-ons Purchased (object) - 75 unique values
    
    📅 DATETIME COLUMNS (0): No datetime columns detected yet
    

## 👀 Step 4: Preview First Rows
**Goal**: Verify content and structure


```python
# Step 4: Preview first rows
print("✅ Step 4 COMPLETED: Data Preview")
print("=" * 60)

print("👀 FIRST 5 ROWS:")
display(df.head())

print("\n🔚 LAST 3 ROWS:")
display(df.tail(3))

print("\n📊 RANDOM SAMPLE (3 rows):")
display(df.sample(3, random_state=42))

print(f"\n📋 COLUMN NAMES ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# Check for any obvious data quality issues in preview
print(f"\n🔍 QUICK DATA QUALITY CHECK:")
print(f"   • All columns present: {'✅' if len(df.columns) == 16 else '❌'}")
print(f"   • No completely empty columns: {'✅' if df.isnull().all().sum() == 0 else '❌'}")
print(f"   • Customer IDs look numeric: {'✅' if df['Customer ID'].dtype in ['int64', 'float64'] else '❌'}")
```

    ✅ Step 4 COMPLETED: Data Preview
    ============================================================
    👀 FIRST 5 ROWS:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Loyalty Member</th>
      <th>Product Type</th>
      <th>SKU</th>
      <th>Rating</th>
      <th>Order Status</th>
      <th>Payment Method</th>
      <th>Total Price</th>
      <th>Unit Price</th>
      <th>Quantity</th>
      <th>Purchase Date</th>
      <th>Shipping Type</th>
      <th>Add-ons Purchased</th>
      <th>Add-on Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>53</td>
      <td>Male</td>
      <td>No</td>
      <td>Smartphone</td>
      <td>SKU1004</td>
      <td>2</td>
      <td>Cancelled</td>
      <td>Credit Card</td>
      <td>5538.33</td>
      <td>791.19</td>
      <td>7</td>
      <td>2024-03-20</td>
      <td>Standard</td>
      <td>Accessory,Accessory,Accessory</td>
      <td>40.21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>53</td>
      <td>Male</td>
      <td>No</td>
      <td>Tablet</td>
      <td>SKU1002</td>
      <td>3</td>
      <td>Completed</td>
      <td>Paypal</td>
      <td>741.09</td>
      <td>247.03</td>
      <td>3</td>
      <td>2024-04-20</td>
      <td>Overnight</td>
      <td>Impulse Item</td>
      <td>26.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>41</td>
      <td>Male</td>
      <td>No</td>
      <td>Laptop</td>
      <td>SKU1005</td>
      <td>3</td>
      <td>Completed</td>
      <td>Credit Card</td>
      <td>1855.84</td>
      <td>463.96</td>
      <td>4</td>
      <td>2023-10-17</td>
      <td>Express</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002</td>
      <td>41</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Smartphone</td>
      <td>SKU1004</td>
      <td>2</td>
      <td>Completed</td>
      <td>Cash</td>
      <td>3164.76</td>
      <td>791.19</td>
      <td>4</td>
      <td>2024-08-09</td>
      <td>Overnight</td>
      <td>Impulse Item,Impulse Item</td>
      <td>60.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1003</td>
      <td>75</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Smartphone</td>
      <td>SKU1001</td>
      <td>5</td>
      <td>Completed</td>
      <td>Cash</td>
      <td>41.50</td>
      <td>20.75</td>
      <td>2</td>
      <td>2024-05-21</td>
      <td>Express</td>
      <td>Accessory</td>
      <td>35.56</td>
    </tr>
  </tbody>
</table>
</div>


    
    🔚 LAST 3 ROWS:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Loyalty Member</th>
      <th>Product Type</th>
      <th>SKU</th>
      <th>Rating</th>
      <th>Order Status</th>
      <th>Payment Method</th>
      <th>Total Price</th>
      <th>Unit Price</th>
      <th>Quantity</th>
      <th>Purchase Date</th>
      <th>Shipping Type</th>
      <th>Add-ons Purchased</th>
      <th>Add-on Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19997</th>
      <td>19996</td>
      <td>27</td>
      <td>Female</td>
      <td>No</td>
      <td>Headphones</td>
      <td>HDP456</td>
      <td>4</td>
      <td>Completed</td>
      <td>Bank Transfer</td>
      <td>1805.90</td>
      <td>361.18</td>
      <td>5</td>
      <td>2024-08-26</td>
      <td>Standard</td>
      <td>Impulse Item, Extended Warranty, Accessory</td>
      <td>198.98</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>19997</td>
      <td>27</td>
      <td>Male</td>
      <td>No</td>
      <td>Headphones</td>
      <td>HDP456</td>
      <td>1</td>
      <td>Cancelled</td>
      <td>Bank Transfer</td>
      <td>2528.26</td>
      <td>361.18</td>
      <td>7</td>
      <td>2024-01-06</td>
      <td>Expedited</td>
      <td>Extended Warranty, Accessory</td>
      <td>101.34</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>19998</td>
      <td>27</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>Laptop</td>
      <td>LTP123</td>
      <td>4</td>
      <td>Completed</td>
      <td>Bank Transfer</td>
      <td>674.32</td>
      <td>674.32</td>
      <td>1</td>
      <td>2024-01-29</td>
      <td>Expedited</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    📊 RANDOM SAMPLE (3 rows):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Loyalty Member</th>
      <th>Product Type</th>
      <th>SKU</th>
      <th>Rating</th>
      <th>Order Status</th>
      <th>Payment Method</th>
      <th>Total Price</th>
      <th>Unit Price</th>
      <th>Quantity</th>
      <th>Purchase Date</th>
      <th>Shipping Type</th>
      <th>Add-ons Purchased</th>
      <th>Add-on Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10650</th>
      <td>11526</td>
      <td>63</td>
      <td>Female</td>
      <td>No</td>
      <td>Laptop</td>
      <td>LTP123</td>
      <td>4</td>
      <td>Cancelled</td>
      <td>PayPal</td>
      <td>5394.56</td>
      <td>674.32</td>
      <td>8</td>
      <td>2024-05-16</td>
      <td>Expedited</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2041</th>
      <td>2805</td>
      <td>35</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Tablet</td>
      <td>SKU1002</td>
      <td>3</td>
      <td>Cancelled</td>
      <td>Debit Card</td>
      <td>1976.24</td>
      <td>247.03</td>
      <td>8</td>
      <td>2024-04-25</td>
      <td>Standard</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8668</th>
      <td>8801</td>
      <td>49</td>
      <td>Female</td>
      <td>No</td>
      <td>Smartwatch</td>
      <td>SKU1003</td>
      <td>3</td>
      <td>Completed</td>
      <td>Paypal</td>
      <td>2534.49</td>
      <td>844.83</td>
      <td>3</td>
      <td>2024-06-07</td>
      <td>Overnight</td>
      <td>Accessory</td>
      <td>19.68</td>
    </tr>
  </tbody>
</table>
</div>


    
    📋 COLUMN NAMES (16 total):
        1. Customer ID
        2. Age
        3. Gender
        4. Loyalty Member
        5. Product Type
        6. SKU
        7. Rating
        8. Order Status
        9. Payment Method
       10. Total Price
       11. Unit Price
       12. Quantity
       13. Purchase Date
       14. Shipping Type
       15. Add-ons Purchased
       16. Add-on Total
    
    🔍 QUICK DATA QUALITY CHECK:
       • All columns present: ✅
       • No completely empty columns: ✅
       • Customer IDs look numeric: ✅
    

## 🔍 Step 5: Check Missing/Null Values
**Goal**: Detect incomplete records early


```python
# Step 5: Check for missing/null values
print("✅ Step 5 COMPLETED: Missing Values Analysis")
print("=" * 60)

# Calculate missing values
missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100

# Create missing values summary
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': missing_percentages.values
}).sort_values('Missing_Count', ascending=False)

print("📊 MISSING VALUES SUMMARY:")
print(missing_summary.to_string(index=False))

# Identify columns with missing data
columns_with_missing = missing_summary[missing_summary['Missing_Count'] > 0]
if not columns_with_missing.empty:
    print(f"\n⚠️  COLUMNS WITH MISSING DATA ({len(columns_with_missing)}):")
    for _, row in columns_with_missing.iterrows():
        print(f"   • {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
else:
    print("\n✅ NO MISSING VALUES DETECTED - Dataset is complete!")

# Check for empty strings or whitespace-only values
print(f"\n🔍 CHECKING FOR EMPTY STRINGS...")
empty_strings_found = False
for col in df.select_dtypes(include=['object']).columns:
    empty_count = (df[col].str.strip() == '').sum()
    if empty_count > 0:
        print(f"   • {col}: {empty_count} empty strings")
        empty_strings_found = True

if not empty_strings_found:
    print("   ✅ No empty strings detected in text columns")

# Overall data completeness
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isnull().sum().sum()
completeness = ((total_cells - missing_cells) / total_cells) * 100

print(f"\n📈 OVERALL DATA COMPLETENESS: {completeness:.2f}%")
print(f"   • Total cells: {total_cells:,}")
print(f"   • Missing cells: {missing_cells:,}")
print(f"   • Complete cells: {total_cells - missing_cells:,}")
```

    ✅ Step 5 COMPLETED: Missing Values Analysis
    ============================================================
    📊 MISSING VALUES SUMMARY:
               Column  Missing_Count  Missing_Percentage
    Add-ons Purchased           4868              24.340
               Gender              1               0.005
                  Age              0               0.000
          Customer ID              0               0.000
         Product Type              0               0.000
                  SKU              0               0.000
               Rating              0               0.000
       Loyalty Member              0               0.000
         Order Status              0               0.000
       Payment Method              0               0.000
           Unit Price              0               0.000
          Total Price              0               0.000
             Quantity              0               0.000
        Purchase Date              0               0.000
        Shipping Type              0               0.000
         Add-on Total              0               0.000
    
    ⚠️  COLUMNS WITH MISSING DATA (2):
       • Add-ons Purchased: 4868 (24.3%)
       • Gender: 1 (0.0%)
    
    🔍 CHECKING FOR EMPTY STRINGS...
       ✅ No empty strings detected in text columns
    
    📈 OVERALL DATA COMPLETENESS: 98.48%
       • Total cells: 320,000
       • Missing cells: 4,869
       • Complete cells: 315,131
    

## 🔄 Step 6: Check for Duplicates
**Goal**: Remove repeated rows if any


```python
# Step 6: Check for duplicate records
print("✅ Step 6 COMPLETED: Duplicate Analysis")
print("=" * 60)

# Check for exact duplicates (all columns identical)
total_duplicates = df.duplicated().sum()
print(f"🔄 EXACT DUPLICATES: {total_duplicates}")

if total_duplicates > 0:
    print(f"   • Duplicate rows found: {total_duplicates}")
    print(f"   • Percentage of duplicates: {(total_duplicates/len(df))*100:.2f}%")
    
    # Show a sample of duplicated rows
    duplicate_rows = df[df.duplicated(keep=False)].sort_values(df.columns.tolist())
    print(f"\n📋 SAMPLE OF DUPLICATE ROWS:")
    display(duplicate_rows.head(6))
    
    # Option to remove duplicates
    print(f"\n⚠️  Consider removing duplicates in preprocessing step")
else:
    print("   ✅ No exact duplicate rows found!")

# Check for potential duplicates based on key identifiers
print(f"\n🔍 CHECKING KEY FIELD DUPLICATES:")

# Check Customer ID + Purchase Date combinations (potential same transaction)
if 'Customer ID' in df.columns and 'Purchase Date' in df.columns:
    customer_date_dups = df.duplicated(subset=['Customer ID', 'Purchase Date']).sum()
    print(f"   • Customer ID + Purchase Date duplicates: {customer_date_dups}")

# Check for multiple transactions per customer
customer_counts = df['Customer ID'].value_counts()
customers_with_multiple = (customer_counts > 1).sum()
max_transactions = customer_counts.max()

print(f"\n👥 CUSTOMER TRANSACTION PATTERNS:")
print(f"   • Unique customers: {df['Customer ID'].nunique():,}")
print(f"   • Customers with multiple transactions: {customers_with_multiple:,}")
print(f"   • Maximum transactions per customer: {max_transactions}")
print(f"   • Average transactions per customer: {customer_counts.mean():.2f}")

# Top customers by transaction count
if customers_with_multiple > 0:
    print(f"\n🏆 TOP 5 CUSTOMERS BY TRANSACTION COUNT:")
    top_customers = customer_counts.head()
    for customer_id, count in top_customers.items():
        print(f"   • Customer {customer_id}: {count} transactions")
```

    ✅ Step 6 COMPLETED: Duplicate Analysis
    ============================================================
    🔄 EXACT DUPLICATES: 0
       ✅ No exact duplicate rows found!
    
    🔍 CHECKING KEY FIELD DUPLICATES:
       • Customer ID + Purchase Date duplicates: 33
    
    👥 CUSTOMER TRANSACTION PATTERNS:
       • Unique customers: 12,136
       • Customers with multiple transactions: 5,499
       • Maximum transactions per customer: 8
       • Average transactions per customer: 1.65
    
    🏆 TOP 5 CUSTOMERS BY TRANSACTION COUNT:
       • Customer 18304: 8 transactions
       • Customer 16357: 7 transactions
       • Customer 7070: 6 transactions
       • Customer 2238: 6 transactions
       • Customer 4224: 6 transactions
    

## 📅 Step 7: Validate Date Parsing
**Goal**: Ensure "Purchase Date" is recognized as datetime


```python
# Step 7: Validate date parsing
print("✅ Step 7 COMPLETED: Date Parsing Validation")
print("=" * 60)

# Check current data type of Purchase Date
print(f"📅 CURRENT PURCHASE DATE INFO:")
print(f"   • Data type: {df['Purchase Date'].dtype}")
print(f"   • Sample values:")
for i, date_val in enumerate(df['Purchase Date'].head(3)):
    print(f"     {i+1}. {date_val}")

# Attempt to parse dates
try:
    # Convert to datetime
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%Y-%m-%d')
    
    print(f"\n✅ DATE PARSING SUCCESSFUL!")
    print(f"   • New data type: {df['Purchase Date'].dtype}")
    
    # Extract date range
    min_date = df['Purchase Date'].min()
    max_date = df['Purchase Date'].max()
    date_range_days = (max_date - min_date).days
    
    print(f"\n📊 DATE RANGE ANALYSIS:")
    print(f"   • Earliest date: {min_date.strftime('%Y-%m-%d (%A)')}")
    print(f"   • Latest date: {max_date.strftime('%Y-%m-%d (%A)')}")
    print(f"   • Date range: {date_range_days} days ({date_range_days/365:.1f} years)")
    
    # Check for any invalid dates or outliers
    current_date = pd.Timestamp.now()
    future_dates = df['Purchase Date'] > current_date
    
    if future_dates.any():
        print(f"⚠️  WARNING: {future_dates.sum()} future dates detected!")
    else:
        print(f"✅ No future dates detected")
        
    # Monthly distribution
    df['Month'] = df['Purchase Date'].dt.month
    df['Year'] = df['Purchase Date'].dt.year
    monthly_counts = df['Purchase Date'].dt.to_period('M').value_counts().sort_index()
    
    print(f"\n📈 MONTHLY DISTRIBUTION:")
    print(f"   • Total months covered: {len(monthly_counts)}")
    print(f"   • Average transactions per month: {monthly_counts.mean():.0f}")
    print(f"   • Peak month: {monthly_counts.idxmax()} ({monthly_counts.max()} transactions)")
    print(f"   • Lowest month: {monthly_counts.idxmin()} ({monthly_counts.min()} transactions)")
    
except Exception as e:
    print(f"❌ DATE PARSING FAILED: {e}")
    print("   • Attempting alternative date formats...")
    
    # Try different date formats
    date_formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y']
    parsed = False
    
    for fmt in date_formats:
        try:
            df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format=fmt, errors='coerce')
            if not df['Purchase Date'].isnull().all():
                print(f"✅ Parsed with format: {fmt}")
                parsed = True
                break
        except:
            continue
    
    if not parsed:
        print("❌ Could not parse dates with common formats")
        print("   • Manual date format investigation required")
```

    ✅ Step 7 COMPLETED: Date Parsing Validation
    ============================================================
    📅 CURRENT PURCHASE DATE INFO:
       • Data type: object
       • Sample values:
         1. 2024-03-20
         2. 2024-04-20
         3. 2023-10-17
    
    ✅ DATE PARSING SUCCESSFUL!
       • New data type: datetime64[ns]
    
    📊 DATE RANGE ANALYSIS:
       • Earliest date: 2023-09-24 (Sunday)
       • Latest date: 2024-09-23 (Monday)
       • Date range: 365 days (1.0 years)
    ✅ No future dates detected
    
    📈 MONTHLY DISTRIBUTION:
       • Total months covered: 13
       • Average transactions per month: 1538
       • Peak month: 2024-01 (2049 transactions)
       • Lowest month: 2023-09 (190 transactions)
    

## 🎯 Phase 1 Summary: Data Overview Complete

**All 7 steps completed successfully!**

✅ **Task Checklist Status:**
- ✅ **Step 1**: Data loaded successfully  
- ✅ **Step 2**: Dataset shape analyzed  
- ✅ **Step 3**: Column types identified  
- ✅ **Step 4**: Data preview completed  
- ✅ **Step 5**: Missing values checked  
- ✅ **Step 6**: Duplicates analyzed  
- ✅ **Step 7**: Date parsing validated  

**Ready for Phase 2: Exploratory Data Analysis (EDA)**

## 📊 Figure Saving Examples

When you create visualizations, use the `save_figure()` function to automatically save them:

```python
# Example 1: Save a histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['Total Price'], bins=30, alpha=0.7)
ax.set_title('Distribution of Total Price')
save_figure(fig, 'price_distribution', category='distributions')

# Example 2: Save correlation heatmap  
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
save_figure(fig, 'correlation_heatmap', category='correlations')

# Example 3: Save ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
save_figure(fig, 'roc_curve_model', category='model_performance')
```

**Figures will be saved as:**
- 📊 `outputs/figures/distributions/price_distribution.png`
- 📊 `outputs/figures/correlations/correlation_heatmap.png` 
- 📊 `outputs/figures/model_performance/roc_curve_model.png`
