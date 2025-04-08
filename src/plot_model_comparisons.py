import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === 1. Model performance data ===
model_results = pd.DataFrame({
    'Model': [
        'Linear Regression', 'Decision Tree', 'Random Forest',
        'AdaBoost', 'GradientBoosting', 'SVR', 'XGBoost'
    ],
    'MSE': [0.036736, 0.061197, 0.031784, 0.046414, 0.031384, 0.087899, 0.030337],
    'RMSE': [0.191668, 0.247381, 0.178279, 0.215438, 0.177157, 0.296477, 0.174176],
    'MAE': [0.098328, 0.061197, 0.063785, 0.110392, 0.067426, 0.177475, 0.065816],
    'RMSLE': [0.134053, 0.171471, 0.125272, 0.154224, 0.123994, 0.207190, 0.121505]
})

# === 2. Convert to long format for grouped bar plot ===
melted = model_results.melt(id_vars='Model', value_vars=['RMSE', 'MAE', 'RMSLE'],
                            var_name='Metric', value_name='Value')

# === 3. Plot grouped bar chart ===
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(data=melted, x='Model', y='Value', hue='Metric', palette='Set2')

plt.title('Model Performance Comparison (RMSE, MAE, RMSLE)')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=30)
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# === 4. Add value labels ===
for bar in ax1.patches:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f'{height:.3f}',
        ha='center',
        va='bottom',
        fontsize=8
    )

# === 5. Save the figure ===
save_path = r"G:\My Drive\french_resturants\outputs\figures\model_comparison_metrics.png"
plt.savefig(save_path, dpi=300)
plt.show()
print(f"✅ Saved figure to: {save_path}")


# === 6. Model comparison (All vs. Top features) ===
comparison_df = pd.DataFrame({
    'Model': ['All Features', 'Top Features'],
    'Train RMSE': [0.1163, 0.1186],
    'Test RMSE': [0.1715, 0.1718]
})
comparison_df['Gap'] = comparison_df['Test RMSE'] - comparison_df['Train RMSE']

comparison_melted = comparison_df.melt(id_vars='Model',
                                       value_vars=['Train RMSE', 'Test RMSE', 'Gap'],
                                       var_name='Metric', value_name='Value')

plt.figure(figsize=(8, 6))
ax2 = sns.barplot(data=comparison_melted, x='Metric', y='Value', hue='Model', palette='Set1')

plt.title('XGBoost: All vs. Top Features Comparison')
plt.ylabel('RMSE / Gap')
plt.xlabel('')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# === 7. Add value labels ===
for bar in ax2.patches:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.002,
        f'{height:.3f}',
        ha='center',
        va='bottom',
        fontsize=8
    )

# === 8. Save figure ===
save_path2 = r"G:\My Drive\french_resturants\outputs\figures\xgboost_feature_comparison.png"
plt.savefig(save_path2, dpi=300)
plt.show()
print(f"✅ Saved XGBoost comparison figure to: {save_path2}")