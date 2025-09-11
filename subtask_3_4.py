import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import pearsonr, spearmanr
import numpy as np


df = pd.read_csv('filtered_WineQT.csv')

class_styles = {
    4: {'color': 'red', 'marker': 'o'},
    6: {'color': 'blue', 'marker': '^'},
    7: {'color': 'green', 'marker': 's'}
}

sns.set_theme(style="whitegrid")

selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']
pairs = list(itertools.combinations(selected_features, 2))

plt.figure(figsize=(12, 10))
for i, (x_feature, y_feature) in enumerate(pairs, 1):
    plt.subplot(len(pairs) // 2 + 1, 2, i)
    for quality in df['quality'].unique():
        subset = df[df['quality'] == quality]
        sns.scatterplot(data=subset, x=x_feature, y=y_feature, 
                        color=class_styles[quality]['color'], 
                        marker=class_styles[quality]['marker'], 
                        label=f'Quality {quality}', s=100)
    plt.title(f'{x_feature} vs {y_feature}')
plt.tight_layout()
plt.savefig('subtask_3_1.png')
plt.close()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, bins=20,
                 kde=True, color='purple', alpha=0.5, 
                 element='poly')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('subtask_3_2.png')
plt.close()




def compute_correlations(data, features, method='pearson'):
    corr_matrix = pd.DataFrame(index=features, columns=features)
    pval_matrix = pd.DataFrame(index=features, columns=features)
    
    for f1, f2 in itertools.combinations(features, 2):
        if method == 'pearson':
            corr, pval = pearsonr(data[f1], data[f2])
        elif method == 'spearman':
            corr, pval = spearmanr(data[f1], data[f2])
        corr_matrix.loc[f1, f2] = corr
        corr_matrix.loc[f2, f1] = corr
        pval_matrix.loc[f1, f2] = pval
        pval_matrix.loc[f2, f1] = pval
    
    # Заполняем диагональ (корреляция признака с самим собой)
    for f in features:
        corr_matrix.loc[f, f] = 1.0
        pval_matrix.loc[f, f] = 0.0
    
    return corr_matrix.astype(float), pval_matrix.astype(float)

pearson_corr, pearson_pval = compute_correlations(df, selected_features, method='pearson')
spearman_corr, spearman_pval = compute_correlations(df, selected_features, method='spearman')

print("\nPearson Correlation Matrix (All Data):")
print(pearson_corr)
print("\nPearson P-values (All Data):")
print(pearson_pval)
print("\nSpearman Correlation Matrix (All Data):")
print(spearman_corr)
print("\nSpearman P-values (All Data):")
print(spearman_pval)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Pearson Correlation (All Data)')
plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Spearman Correlation (All Data)')
plt.tight_layout()
plt.savefig('correlation_heatmap_all.png')
plt.close()

for quality in df['quality'].unique():
    subset = df[df['quality'] == quality]
    pearson_corr, pearson_pval = compute_correlations(subset, selected_features, method='pearson')
    spearman_corr, spearman_pval = compute_correlations(subset, selected_features, method='spearman')
    
    print(f"\nPearson Correlation Matrix (Quality {quality}):")
    print(pearson_corr)
    print(f"\nPearson P-values (Quality {quality}):")
    print(pearson_pval)
    print(f"\nSpearman Correlation Matrix (Quality {quality}):")
    print(spearman_corr)
    print(f"\nSpearman P-values (Quality {quality}):")
    print(spearman_pval)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Pearson Correlation (Quality {quality})')
    plt.subplot(1, 2, 2)
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Spearman Correlation (Quality {quality})')
    plt.tight_layout()
    plt.savefig(f'correlation_heatmap_quality_{quality}.png')
    plt.close()
