import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import pearsonr, spearmanr

%matplotlib inline

# Загрузка и фильтрация датасета
df = pd.read_csv('WineQT.csv')

selected_classes = [4, 6, 7]
filtered_df = df[df['quality'].isin(selected_classes)]
selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'Id']
filtered_df = filtered_df.dropna(subset=selected_features)
filtered_df = filtered_df[selected_features + ['quality']]

filtered_df.to_csv('filteredWtID_WineQT.csv', index=False)
print('filtered DataFrame saved to: filteredWtID_WineQT.csv')

# Загрузка отфильтрованного датасета
df = pd.read_csv('filteredWtID_WineQT.csv')

# Оставляем только нужные признаки
selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'Id']
df = df[selected_features + ['quality']]  # Сохраняем quality для группировки

class_styles = {
    4: {'color': 'red', 'marker': 'o'},
    6: {'color': 'blue', 'marker': '^'},
    7: {'color': 'green', 'marker': 's'}
}

sns.set_theme(style="whitegrid")

pairs = list(itertools.combinations(selected_features, 2))

# Нормализация данных, чтобы не было диспропорциональных сторон по типу
# citric acid 0.0 0.2 0.4 0.6 0.8 1.0 && residual sugar 5 10 15
df_norm = df.copy()
for feature in selected_features:
    df_norm[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

# Диаграммы рассеяния
plt.figure(figsize=(12, 12), constrained_layout=True)
for i, (x_feature, y_feature) in enumerate(pairs, 1):
    ax = plt.subplot(4, 3, i)
    for quality in df['quality'].unique():
        subset = df_norm[df_norm['quality'] == quality]
        sns.scatterplot(data=subset, x=x_feature, y=y_feature, 
                        color=class_styles[quality]['color'], 
                        marker=class_styles[quality]['marker'], 
                        label=f'Quality {quality}', s=50)
    ax.set_aspect('equal')  # Квадратное соотношение сторон
    ax.set_xlim(-0.1, 1.1)  # Унифицированный масштаб 0–1
    ax.set_ylim(-0.1, 1.1)
    plt.title(f'{x_feature} vs {y_feature}')

# Удаление пустых subplot'ов
for j in range(len(pairs), len(plt.gcf().axes)):
    plt.delaxes(plt.gcf().axes[j])
plt.show()

# Гистограммы распределения признаков
plt.figure(figsize=(12, 12), constrained_layout=True)
for i, feature in enumerate(selected_features, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=df_norm, x=feature, bins=20, color='purple', alpha=0.5, 
                 element='poly', stat='density')
    sns.kdeplot(data=df_norm, x=feature, color='purple', alpha=0.7, bw_adjust=0.5)
    plt.title(f'Distribution of {feature} (Normalized)')
    plt.xlim(-0.1, 1.1)
    plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio())
# Удаление пустых subplot'ов
for j in range(len(selected_features), len(plt.gcf().axes)):
    plt.delaxes(plt.gcf().axes[j])
plt.show()

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
print(pearson_corr.round(4))
print("\nPearson P-values (All Data):")
print(pearson_pval.round(4))
print("\nSpearman Correlation Matrix (All Data):")
print(spearman_corr.round(4))
print("\nSpearman P-values (All Data):")
print(spearman_pval.round(4))

# Тепловые карты корреляций для всего датасета
plt.figure(figsize=(8, 8), constrained_layout=True)  # Квадратная для 5x5
plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f', annot_kws={'size': 8})
plt.title('Pearson Correlation (All Data)')
plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f', annot_kws={'size': 8})
plt.title('Spearman Correlation (All Data)')
plt.show()

# Вычисление и вывод корреляций для каждого класса
for quality in df['quality'].unique():
    subset = df[df['quality'] == quality]
    pearson_corr, pearson_pval = compute_correlations(subset, selected_features, method='pearson')
    spearman_corr, spearman_pval = compute_correlations(subset, selected_features, method='spearman')
    
    print(f"\n{'='*50}")
    print(f"Pearson Correlation Matrix (Quality {quality}):")
    print(pearson_corr.round(4))
    print(f"\nPearson P-values (Quality {quality}):")
    print(pearson_pval.round(4))
    print(f"\nSpearman Correlation Matrix (Quality {quality}):")
    print(spearman_corr.round(4))
    print(f"\nSpearman P-values (Quality {quality}):")
    print(spearman_pval.round(4))
    
    # Тепловые карты для каждого класса
    plt.figure(figsize=(8, 8), constrained_layout=True)
    plt.subplot(1, 2, 1)
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f', annot_kws={'size': 8})
    plt.title(f'Pearson Correlation (Quality {quality})')
    plt.subplot(1, 2, 2)
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f', annot_kws={'size': 8})
    plt.title(f'Spearman Correlation (Quality {quality})')
    plt.show()
    










import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import pearsonr, spearmanr

%matplotlib inline

df = pd.read_csv('filtered_WineQT.csv')

class_styles = {
    4: {'color': 'red', 'marker': 'o'},
    6: {'color': 'blue', 'marker': '^'},
    7: {'color': 'green', 'marker': 's'}
}

sns.set_theme(style="whitegrid")

selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']

# Нормализация данных, чтобы не было диспропорциональных сторон по типу
# citric acid 0.0 0.2 0.4 0.6 0.8 1.0 && residual sugar 5 10 15
df_norm = df.copy()
for feature in selected_features:
    df_norm[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

# Матрица 4x4: рассеяние вне диагонали, гистограммы с KDE на диагонали
plt.figure(figsize=(12, 12), constrained_layout=True)
for i, x_feature in enumerate(selected_features):
    for j, y_feature in enumerate(selected_features):
        ax = plt.subplot(4, 4, i * 4 + j + 1)
        if i == j:  # Диагональ: гистограммы с KDE
            sns.histplot(data=df_norm, x=x_feature, bins=20, color='purple', alpha=0.5, 
                         element='poly', stat='density')
            sns.kdeplot(data=df_norm, x=x_feature, color='purple', alpha=0.7, bw_adjust=0.5)
            plt.title(f'Distribution of {x_feature} (Normalized)')
            plt.xlim(-0.1, 1.1)
            plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio())
        else:  # Вне диагонали: диаграммы рассеяния
            for quality in df['quality'].unique():
                subset = df_norm[df_norm['quality'] == quality]
                sns.scatterplot(data=subset, x=x_feature, y=y_feature, 
                                color=class_styles[quality]['color'], 
                                marker=class_styles[quality]['marker'], 
                                label=f'Quality {quality}', s=50, alpha=0.1)
            ax.set_aspect('equal')  # Квадратное соотношение сторон
            ax.set_xlim(-0.1, 1.1)  # Унифицированный масштаб 0–1
            ax.set_ylim(-0.1, 1.1)
            plt.title(f'{x_feature} vs {y_feature}')
plt.show()

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
print(pearson_corr.round(4))
print("\nPearson P-values (All Data):")
print(pearson_pval.round(4))
print("\nSpearman Correlation Matrix (All Data):")
print(spearman_corr.round(4))
print("\nSpearman P-values (All Data):")
print(spearman_pval.round(4))

# Тепловые карты корреляций для всего датасета
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f')
plt.title('Pearson Correlation (All Data)')
plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f')
plt.title('Spearman Correlation (All Data)')
plt.tight_layout()
plt.show()

# Вычисление и вывод корреляций для каждого класса
for quality in df['quality'].unique():
    subset = df[df['quality'] == quality]
    pearson_corr, pearson_pval = compute_correlations(subset, selected_features, method='pearson')
    spearman_corr, spearman_pval = compute_correlations(subset, selected_features, method='spearman')
    
    print(f"\n{'='*50}")
    print(f"Pearson Correlation Matrix (Quality {quality}):")
    print(pearson_corr.round(4))
    print(f"\nPearson P-values (Quality {quality}):")
    print(pearson_pval.round(4))
    print(f"\nSpearman Correlation Matrix (Quality {quality}):")
    print(spearman_corr.round(4))
    print(f"\nSpearman P-values (Quality {quality}):")
    print(spearman_pval.round(4))
    
    # Тепловые карты для каждого класса
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f')
    plt.title(f'Pearson Correlation (Quality {quality})')
    plt.subplot(1, 2, 2)
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.4f')
    plt.title(f'Spearman Correlation (Quality {quality})')
    plt.tight_layout()
    plt.show()