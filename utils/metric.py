import numpy as np

def accuracy_score(y_true, y_pred):
    """
    计算准确率
    """
    return np.sum(y_true == y_pred) / len(y_true)
def roc_auc_score(y_true, y_scores):
    """
    计算 ROC AUC
    """
    # 对预测分数排序，计算真正率 (TPR) 和假阳性率 (FPR)
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # 使用梯形法则计算AUC
    auc = np.trapz(tpr_list, fpr_list)
    return auc
def f1_score(y_true, y_pred):
    """
    计算 F1-score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
def auc(precision_list, recall_list):
    """
    计算 AUPR
    """
    return np.trapz(precision_list, recall_list)  # 使用梯形法则计算AUPR
def precision_score(y_true, y_pred):
    """
    计算精确率
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    return tp / (tp + fp) if (tp + fp) != 0 else 0
def recall_score(y_true, y_pred):
    """
    计算召回率
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp / (tp + fn) if (tp + fn) != 0 else 0
def precision_recall_curve(y_true, y_prob):
    """
    计算不同阈值下的精确率和召回率
    返回 (precision_list, recall_list, thresholds)
    """
    # 排序预测概率和真实标签
    thresholds = np.sort(np.unique(y_prob))[::-1]  # 从大到小
    precision_list = []
    recall_list = []
    # 遍历所有阈值
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
    return precision_list, recall_list, thresholds
def roc_curve(y_true, y_prob):
    """
    计算 ROC 曲线的假阳性率 (FPR) 和 真正率 (TPR)
    返回 (fpr, tpr, thresholds)
    """
    # 排序预测概率
    thresholds = np.sort(np.unique(y_prob))[::-1]  # 从大到小
    fpr_list = []
    tpr_list = []
    # 计算 FPR 和 TPR
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))  # 真正例
        fp = np.sum((y_true == 0) & (y_pred == 1))  # 假正例
        fn = np.sum((y_true == 1) & (y_pred == 0))  # 假负例
        tn = np.sum((y_true == 0) & (y_pred == 0))  # 真负例
        # 计算 TPR (真正率) 和 FPR (假阳性率)
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # TPR = TP / (TP + FN)
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # FPR = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list, thresholds
def cohen_kappa_score(y_true, y_pred):
    """
    计算Kappa系数
    """
    n = len(y_true)
    observed_agreement = np.sum(y_true == y_pred) / n
    expected_agreement = ((np.sum(y_true == 1) / n) * (np.sum(y_pred == 1) / n)) + ((np.sum(y_true == 0) / n) * (np.sum(y_pred == 0) / n))
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)
def matthews_corrcoef(y_true, y_pred):
    """
    计算Matthews相关系数
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / denominator if denominator != 0 else 0

def mean_absolute_error(y_true, y_pred):
    """
    计算平均绝对误差 (MAE)
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    计算均方误差 (MSE)
    """
    return np.mean((y_true - y_pred) ** 2)