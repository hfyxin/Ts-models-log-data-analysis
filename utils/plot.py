# This script speicifies all plot functions for notebook and presentation purpose.
import matplotlib.pyplot as plt

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 16

def dual_train_curve(train_loss, val_loss, train_acc, val_acc,
        figsize=(16,5), legend=True, tick_interval=False):
    '''Loss & Accuracy during training. Plot 2 graphs side by side.
    
    Parameters:
    -----------
    legend: toggle legend ON/OFF.

    tick_interval: when epochs is small, the plot will show float ticks.
    Specify this parameter to show all integer ticks.
    '''
    
    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    if legend: plt.legend(['train loss', 'val los'], fontsize=LEGEND_SIZE)
    plt.xlabel('epochs', fontsize=LABEL_SIZE)
    plt.ylabel('loss', fontsize=LABEL_SIZE)
    plt.title('Training loss', fontsize=TITLE_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(train_acc)
    plt.plot(val_acc)
    if legend: plt.legend(['train accuracy', 'val accuracy'], fontsize=LEGEND_SIZE)
    plt.xlabel('epochs', fontsize=LABEL_SIZE)
    plt.ylabel('accuracy', fontsize=LABEL_SIZE)
    plt.title('Training accuracy', fontsize=TITLE_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.ylim((0.4,0.95))
    plt.grid()


def dual_pr_curve(pr1, rc1, pr2, rc2,
        figsize=(12,5), show_text=False):
    '''Precision-Recall curve. Plot 2 graphs. Used in converted PR plots.'''

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.plot(pr1, rc1)
    plt.xlabel('Precision', fontsize=LABEL_SIZE)
    plt.ylabel('Recall', fontsize=LABEL_SIZE)
    plt.xlim((-0.05,1.05))
    plt.title('P-R Curve', fontsize=TITLE_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(pr2, rc2)
    plt.xlabel('Precision', fontsize=LABEL_SIZE)
    # plt.ylabel('Recall', fontsize=14)
    plt.xlim((-0.05,1.05))
    plt.title('Converted P-R Curve', fontsize=TITLE_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid()