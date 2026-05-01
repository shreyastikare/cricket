import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, brier_score_loss, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette = 'deep')
sns.set_context("notebook", font_scale=1)
sns.despine(trim=True)

plt.rcParams.update({
    'font.size': 10, 
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8, 
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    'figure.dpi': 150,
    'savefig.dpi': 300,
})
plt.close()


def discrim(log_model, X, y):
    p_hat = log_model.predict_proba(X)[:, 1]
    p = pd.DataFrame({
        'p_hat': p_hat,
        'y': y
    })
    
    p0 = p[p['y'] == 0]['p_hat']
    p1 = p[p['y'] == 1]['p_hat']
    
    coef_discrim = p1.mean() - p0.mean()
    
    # fig, axes = plt.subplots(figsize = (10,10))
    sns.kdeplot(data = p, x = 'p_hat', hue = 'y', common_norm = False, fill = True)
    plt.title(f'Coefficient of discrimination = {coef_discrim:.4f}')
    plt.xlabel('Predicted Probability')
    plt.xlim(0,1)
    plt.ylabel('Density')
    plt.show()
    
    return


def roc_auc(log_model, X, y):
    p_hat = log_model.predict_proba(X)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, p_hat)
    
    RocCurveDisplay(fpr = fpr, tpr = tpr).plot()
    plt.title(f'ROC curve, AUC = {auc(fpr, tpr):.4f}')
    
    plt.show()
    
    return


def glm_summary(log_pipeline, X, y):
    # coefficient of discrimination
    p_hat = log_pipeline.predict_proba(X)[:, 1]
    p = pd.DataFrame({
        'p_hat': p_hat,
        'y': y
    })
    
    p0 = p[p['y'] == 0]['p_hat']
    p1 = p[p['y'] == 1]['p_hat']
    coef_discrim = p1.mean() - p0.mean()
    
    fpr, tpr, thresholds = roc_curve(y, p_hat)
    ks_values = tpr - fpr
    
    ks_max_idx = np.argmax(ks_values)
    ks_max = ks_values[ks_max_idx]
    ks_cutoff = thresholds[ks_max_idx]
    
    # accuracy
    y_pred = (p_hat > ks_cutoff).astype(int)
    accuracy = accuracy_score(y, y_pred) * 100
    
    # brier score
    brier_score = brier_score_loss(y_true = y, y_proba = p_hat)
    print(f'            best cutoff : {ks_cutoff:.4f}\n')
    print(f'    validation accuracy : {accuracy:.2f} %')
    print(f'         validation AUC : {auc(fpr, tpr):.4f}')
    print(f'           KS Statistic : {ks_max:.4f}')
    print(f' validation Brier score : {brier_score:.4f}')
    print(f'Coef. of discrimination : {coef_discrim:.4f}\n')
    
    
    print(f'   Regularzation lambda : {log_pipeline.named_steps["model"].C_[0]:.4f}')
    print(f'               l1 ratio : {log_pipeline.named_steps["model"].l1_ratio_[0]:.4f}\n')
    
    # non-zero features ordered by decreasing magnitude
    feature_names = log_pipeline.named_steps['preprocess'].get_feature_names_out()
    raw_coef = log_pipeline['model'].coef_[0]
    
    features = feature_names[raw_coef != 0]
    coef = raw_coef[raw_coef != 0]
    
    coef_df = (
        pd.DataFrame({
            'feature': features,
            'coef': coef
        })
        .assign(abs_coef = lambda df: df['coef'].abs())
        .sort_values('abs_coef', ascending = False)
        .reset_index(drop =  True)
        .drop(columns = ['abs_coef'])
    )
    print(coef_df)
    
    return ks_cutoff


def accuracy_by_over(log_model, X, cutoff = 0.5, bin_width = 3, min_over = 0, max_over = 20):
    overs = range(min_over, max_over - bin_width + 1)
    accuracy_overs = []
    for i in overs:
        X_temp = X[(X['team_balls'] > i * 6) & (X['team_balls'] <= ((i + bin_width) * 6))]
        y_temp = X_temp['bowling_team_won'].to_numpy()
        y_prob_temp = log_model.predict_proba(X_temp)[:, 1]
        y_pred_temp = (y_prob_temp >= cutoff).astype(int)
        accuracy_temp = accuracy_score(y_temp, y_pred_temp) * 100
        accuracy_overs.append(accuracy_temp)

    fig, axes = plt.subplots()
    sns.lineplot(x = overs, y = accuracy_overs, ax = axes)
    plt.xlabel('Overs')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy by Over (Validation)')
    plt.plot()


def phase_metrics(X, yt, yp, yt_full, yp_full, splits = [6, 10, 16, 20]):
    Xt = (X.copy()).reset_index(drop = True)
    if splits[0] >= 6:
        Xt['powerplay_balls_remaining'] = 0
    Xt['team_balls'] = 120 - Xt['powerplay_balls_remaining'] - Xt['middle_balls_remaining'] - Xt['death_balls_remaining']
    n = len(splits)
    phase_start = []
    phase_end = []
    mae = []
    mape = []
    std = []
    
    phase_start.append(splits[0])
    phase_end.append(splits[-1])
    
    mae_full = mean_absolute_error(yt, yp)
    mae.append(mae_full)
    
    std_full = np.std(yt - yp)
    std.append(std_full)
    
    mape_full = mean_absolute_percentage_error(yt_full, yp_full) * 100
    mape.append(mape_full)
    
    for i in range(n-1):
        phase_start.append(splits[i])
        phase_end.append(splits[i + 1])
        test_idx = Xt.index[(Xt['team_balls'] > (splits[i] * 6)) & (Xt['team_balls'] <= (splits[i + 1] * 6))]
        y_test_phase = yt[test_idx]
        y_pred_phase = yp[test_idx]
        y_test_full_phase = yt_full[test_idx]
        y_pred_full_phase = yp_full[test_idx]
        
        mae_phase = mean_absolute_error(y_test_phase, y_pred_phase)
        mae.append(mae_phase)
        
        std_phase = np.std(y_test_phase - y_pred_phase)
        std.append(std_phase)
        
        mape_phase = mean_absolute_percentage_error(y_test_full_phase, y_pred_full_phase) * 100
        mape.append(mape_phase)
    
    metrics = pd.DataFrame({
        'phase_start': phase_start,
        'phase_end': phase_end,
        'mae': mae,
        'mape': mape,
        'std': std
    })
    
    return(metrics)