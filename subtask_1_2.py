import pandas as pd


df = pd.read_csv('WineQT.csv')

dims = df.shape

num_features = len(df.columns) - 2  # Исключаем 'quality' (целевая) и 'Id'

class_counts = df['quality'].value_counts().sort_index()
num_classes = len(class_counts)

missing_values = df.isnull().sum().sum()
percent_missing = (missing_values / (df.shape[0] * df.shape[1])) * 100

feature_types = df.dtypes
quality_mean = df['quality'].mean()
quality_std = df['quality'].std()
duplicates = df.duplicated().sum()

print(f"Dimensionality of the dataset: {dims[0]} rows × {dims[1]} columns")
print(f"Number of features: {num_features}")
print(f"Number of target classes: {num_classes} (quality scores: {', '.join(map(str, class_counts.index))})")
print(f"Objects in each class: {', '.join(f'{k}: {v}' for k, v in class_counts.items())}")
print(f"Percentage of objects with undefined features: {percent_missing:.2f}%")
print("Other key characteristics:" + 
      f"\n- All features types: {', '.join(f'{k}: {v}' for k, v in feature_types.items())}" +
      f"\n- Quality ranges from {df['quality'].min()} to {df['quality'].max()} with mean {quality_mean:.2f} and std {quality_std:.2f}" +
      f"\n- Duplicates: {duplicates}")

selected_classes = [4, 6, 7]
filtered_df = df[df['quality'].isin(selected_classes)]
selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']
filtered_df = filtered_df.dropna(subset=selected_features)
filtered_df = filtered_df[selected_features + ['quality']]

filtered_df.to_csv('filtered_WineQT.csv', index=False)

print('filtered DataFrame saved to: filtered_WineQT.csv')
