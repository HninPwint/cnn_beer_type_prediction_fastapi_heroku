import sklearn.metrics as metr
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List
from sklearn.calibration import calibration_curve
from pathlib import Path, WindowsPath
from dotenv import find_dotenv

project_dir = Path(find_dotenv()).parent
reports_dir = project_dir / 'reports'


def classification_reports(classifier, X, y, verbose=False):
    """
    Retrieved from: https://www.kaggle.com/asiyehbahaloo/asiyeh-bahaloo
    
    Provides Confusion matrix, accuracy, AUC and standard classification report. Note Verbose returns ROC curve values and accuracy
    
    """
    y_pred = classifier.predict_proba(X)[:, 1]
    y_pred_lab = classifier.predict(X)
    size_data = len(y)
    count_class_1 = sum(y)
    count_class_0 = size_data - count_class_1
    print(' class 1 : ', count_class_1)
    print(' class 0 : ', count_class_0)
    fpr, tpr, thresholds = metr.roc_curve(y, y_pred)
    print("Confusion Matrix: \n", metr.confusion_matrix(y, y_pred_lab))
    score = metr.accuracy_score(y, y_pred_lab)
    print("Accuracy: ", score)
    auc = metr.roc_auc_score(y, y_pred)
    print("AUC: ", auc)
    print(metr.classification_report(y, y_pred_lab))
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange')
    plt.show()
    if verbose:
        return fpr, tpr, score


def plot_pie(y):
    labels = 'Positive', 'Negative'
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    sizes = [pos_count, neg_count]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis(
        'equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def classification_reports_2(y, preds, probs, verbose=False):
    """
    Modification of classification_reports(). This function accepts predictions,
    instead of conducting the prediction. This is useful because there are
    other parameters used in the predict method that can't be passed into
    classification_reports().
    """

    size_data = len(y)
    count_class_1 = sum(y)
    count_class_0 = size_data - count_class_1
    print(' class 1 : ', count_class_1)
    print(' class 0 : ', count_class_0)
    fpr, tpr, thresholds = metr.roc_curve(y, probs)
    print("Confusion Matrix: \n", metr.confusion_matrix(y, preds))
    score = metr.accuracy_score(y, preds)
    print("Accuracy: ", score)
    auc = metr.roc_auc_score(y, probs)
    print("AUC: ", auc)
    print(metr.classification_report(y, preds))
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange')
    plt.show()
    if verbose:
        return fpr, tpr, score


def create_feature_importance_df(
        clf,
        feature_names: List[str],
        importance_types: List[str] = ['gain', 'cover', 'weight']
) -> pd.DataFrame:
    """
    Convenience function to extract the importance values from the XGB
    Classifier object
    :param clf: an XGB Classifier object
    :param feature_names: a list of human readable column names
    :param importance_types: a list of the different types of importances to calculate
    :return: pd.DataFrame
    """
    importance_df = pd.DataFrame()

    # The XGB feature names are not interpretable
    mapping_dict = dict(zip(clf.get_booster().feature_names, feature_names))

    for importance_type in importance_types:
        df = pd.DataFrame()
        importance_dict = clf.get_booster().get_score(
            importance_type=importance_type)
        df['feature'] = importance_dict.keys()
        df['score'] = importance_dict.values()
        df['importance_type'] = importance_type

        importance_df = pd.concat([importance_df, df])

    importance_df.loc[:, 'feature'] = importance_df.feature.map(mapping_dict)

    return importance_df


def plot_feature_importances(df: pd.DataFrame,
                             top_n: int = 30,
                             order_by: str = 'gain'):
    """
    :param df: a df of the format returned by `create_feature_importance_df`
    :param top_n: take the top n features only
    :param order_by: the chart will be sorted by one of the importance types
    :return:
    """

    order = (
        df
            .query('importance_type == @order_by')
            .sort_values(by='score', ascending=False)
            .iloc[:top_n, 0]
    )

    ax = sns.catplot(x='score',
                     y='feature',
                     order=order,
                     kind='bar',
                     orient='horizontal',
                     data=df,
                     color='Grey',
                     col='importance_type',
                     sharex=False)


def plot_calibration_curve(y_true, probs, n_bins=5, normalize=False):
    """
    """
    prob_true, prob_pred = calibration_curve(y_true,
                                             probs,
                                             n_bins=n_bins,
                                             normalize=normalize)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(prob_true, prob_pred, marker='.')
    plt.show()


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    From https://deeplizard.com/learn/video/0LhiS6yu2qQ
    :param cm:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')