import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import math
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import numpy as np
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Функция для построения гистограммы распределения признаков и вывода стастистики
def print_stats(dataframe, feature_name, k):
    plt.figure(figsize=(6, 4))
    plt.hist(dataframe[feature_name], bins=k, edgecolor='black')
    plt.title(f'Распределение признака {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.show()

    shapiro_test = stats.shapiro(dataframe[feature_name])
    dagostino_test = stats.normaltest(dataframe[feature_name])
    ks_test = stats.kstest(
        dataframe[feature_name],
        'norm',
        args=(dataframe[feature_name].mean(), dataframe[feature_name].std())
    )

    def interpret_pvalue(p):
        if p > 0.05:
            return "Нормальное"
        elif p > 0.01:
            return "Близко к нормальному"
        else:
            return "Далеко от нормального"
    

    # Дескриптивный анализ
    print (
        f"Статистика для признака {feature_name}:\n"
        f"--------------------------|-------------------------------------------\n"
        f"Среднее:                  | {dataframe[feature_name].mean()}\n"
        f"Медиана:                  | {dataframe[feature_name].median()}\n"
        f"Мода:                     | {dataframe[feature_name].mode()[0]}\n"
        f"Минимум:                  | {dataframe[feature_name].min()}\n"
        f"Максимум:                 | {dataframe[feature_name].max()}\n"
        f"Размах:                   | {dataframe[feature_name].max() - dataframe[feature_name].min()}\n"
        f"Стандартное отклонение:   | {dataframe[feature_name].std()}\n"
        f"Дисперсия:                | {dataframe[feature_name].var()}\n"
        f"Асимметрия:               | {dataframe[feature_name].skew()}\n"
        f"Эксцесс:                  | {dataframe[feature_name].kurtosis()}\n"
        f"Квантили (25%, 50%, 75%): | {dataframe[feature_name].quantile([0.25, 0.5, 0.75]).to_dict()}\n"
        f"Оценка нормальности распредления:\n"
        f"--------------------------|-------------------------------------------\n"
        f"Шапиро-Уилк:              | p={shapiro_test.pvalue:.3e} → {interpret_pvalue(shapiro_test.pvalue)}\n"
        f"Д'Агостино-Пиросона:      | p={dagostino_test.pvalue:.3e} → {interpret_pvalue(dagostino_test.pvalue)}\n"
        f"Колмагорова-Смирнова      | p={ks_test.pvalue:.3e} → {interpret_pvalue(ks_test.pvalue)}\n"
    )

# Загрузка данных
file_path = "concrete_data.csv"

df = kagglehub.dataset_load(
  kagglehub.KaggleDatasetAdapter.PANDAS,
  "elikplim/concrete-compressive-strength-data-set",
  file_path,
)

n = df.shape[0]
k = int(math.ceil(1 + math.log2(n)))
print(f"Количество записей: {n}")
print(f"Количество интервалов по формуле Стержиса: {k}")
print(f"Количество пропущенных значений: {df.isnull().sum().sum()}")

# Дискрпетивный анализ
for feature_name in ["cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age", "concrete_compressive_strength"]:
    print_stats(df, feature_name, k)


x = df.drop(columns=['concrete_compressive_strength'])
y = df['concrete_compressive_strength']

# Стандартизация признаков
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#  Критерий КМО
kmo_all, kmo_model = calculate_kmo(x_scaled)
print("KMO по переменным:")
print(pd.Series(kmo_all, index=x.columns))
print("\nKMO по модели (общий показатель):", round(kmo_model, 3))

# Критерий Бартлетта
chi_square_value, p_value = calculate_bartlett_sphericity(x_scaled)
print("\nКритерий Бартлетта:")
print("Chi-square =", round(chi_square_value, 3))
print("p-value    =", p_value)

# Построение матрицы корреляции
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='PiYG', vmin=-1, vmax=1)
plt.title("Матрица корреляции признаков", fontsize=14)
plt.show()


# Определение факторов методом главных компонент. Критерий Кайзера
feature_cols = df.drop(columns=["concrete_compressive_strength"]).columns.tolist()

pca = PCA()
pca.fit(x_scaled)

eigenvalues = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_
cum_variance_ratio = np.cumsum(variance_ratio)

pca_components = pd.DataFrame(
    pca.components_,                
    columns=df.drop(columns=['concrete_compressive_strength']).columns,        
    index=[f'PC{i+1}' for i in range(pca.n_components_)]  
)

print(pca_components.round(3))

pca_table = pd.DataFrame(
    { "Собственное значение (λ)": eigenvalues,},
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print(pca_table)

n_factors_kaiser = np.sum(eigenvalues > 1)
print(f"Число факторов по критерию Кайзера (λ > 1): {n_factors_kaiser}")


# Факторные нагрузки до вращения
feature_cols = df.drop(columns=["concrete_compressive_strength"]).columns.tolist()
n_factors = n_factors_kaiser
fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='principal')
fa.fit(x_scaled)  

loadings_no_rot = pd.DataFrame(
    fa.loadings_,
    index=feature_cols,
    columns=[f"Factor{i+1}" for i in range(n_factors)]
)

plt.figure(figsize=(8, 5))
sns.heatmap(loadings_no_rot,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0)
plt.title("Факторные нагрузки до вращения")
plt.yticks(rotation=0)
plt.show()


# Факторные нагрузки после вращения
fa_rot = FactorAnalyzer(n_factors=n_factors, method='principal', rotation='varimax')
fa_rot.fit(x_scaled)

loadings_rot = pd.DataFrame(
    fa_rot.loadings_,
    index=feature_cols,
    columns=[f"Factor{i+1}" for i in range(n_factors)]
)


plt.figure(figsize=(8, 5))
sns.heatmap(
    loadings_rot,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Факторные нагрузки после вращения")
plt.yticks(rotation=0)
plt.show()

# Диаграммы рассеивания
factor_scores = fa_rot.transform(x_scaled)
n_factors = factor_scores.shape[1]
factor_cols = [f"Factor{i+1}" for i in range(n_factors)]

scores_df = pd.DataFrame(factor_scores, columns=factor_cols)
scores_df["concrete_compressive_strength"] = df["concrete_compressive_strength"].values

pairs = [
    ("Factor2", "Factor1"),
    ("Factor3", "Factor1"),
    ("Factor4", "Factor1"),
    ("Factor3", "Factor2"),
    ("Factor4", "Factor2"),
    ("Factor4", "Factor3"),
]

strength = scores_df["concrete_compressive_strength"].values
norm = Normalize(vmin=strength.min(), vmax=strength.max())
cmap = cm.get_cmap("plasma") 
colors = cmap(norm(strength))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, (fx, fy) in zip(axes.ravel(), pairs):
    ax.scatter(
        scores_df[fx],
        scores_df[fy],
        c=colors,
        s=15,
        alpha=0.7
    )
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title(f"{fy} vs {fx}")
    ax.grid(True, linestyle=":", alpha=0.5)

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Concrete compressive strength")

plt.suptitle("Диаграммы рассеяния факторных баллов\n(окраска по прочности бетона)", y=0.98, fontsize=14)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()


# Метод каменистой осыпи
eigenvalues = np.array(eigenvalues)
factors = np.arange(1, len(eigenvalues) + 1)
kaiser_count = 4
elbow_factor = 4

plt.figure(figsize=(12, 6))
plt.plot(factors, eigenvalues, marker='o', linestyle='-', color='b', label='Собственные значения')
plt.axhline(1, color='r', linestyle='--', label='Критерий Кайзера (λ=1)')
plt.axvline(elbow_factor, color='green', linestyle='-', label=f'Точка "локтя" (фактор {elbow_factor})')
plt.axvline(kaiser_count, color='orange', linestyle='--',
            label=f'Критерий Кайзера (фактор {kaiser_count})')

plt.title("График каменистой осыпи\nдля определения оптимального количества факторов")
plt.xlabel("Номер фактора")
plt.ylabel("Собственное значение")
plt.grid(True, linestyle=':')
plt.legend(loc='best')

plt.tight_layout()
plt.show()