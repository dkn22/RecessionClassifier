def plot_confusion_matrix(cnf_matrix, class_labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    import itertools
        
    fig, ax = plt.subplots(nrows=1)
    ax.imshow(cnf_matrix, interpolation='nearest', cmap=cmap, alpha=0)
    #plt.axis('off')
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    ax.set_xticklabels(class_labels, fontsize=16)
    ax.set_yticklabels(class_labels, fontsize=16)
    
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.annotate(str(cnf_matrix[i, j]), xy=(j, i), horizontalalignment='center')
        
        
    ax.set_ylabel('True state', fontsize=14)
    ax.set_xlabel('Predicted state', fontsize=14)
    # ax.set_title(title, fontsize=20)
    ax.grid(False)
    

def plot_roc_curve(y_test, y_pred_proba, classifier_label):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
    roc_auc = auc(fpr, tpr)
    
    figure, ax = plt.subplots(1,1)
    
    # if multiple_plots:
    
    ax.plot(fpr, tpr, lw=1, label = classifier_label + ' (AUC = %.2f)' %(roc_auc))
    
    label_kwargs = {}
    label_kwargs['bbox'] = dict(
                boxstyle='round,pad=0.3', fc='darkorange', alpha=0.5,
            )
    
#     for thres in [0.3, 0.5, 0.6]:
#         idx = np.where(np.round(thresholds, 2) == thres)[0][0]
#         threshold = np.round(thresholds[idx], 2)
#         ax.annotate('Threshold = ' + str(thres), xy=(fpr[idx], tpr[idx]),
#                 xytext=(fpr[idx]-0.12, tpr[idx]+0.06),
#                 arrowprops=dict(facecolor='gray', shrink=5, headwidth=2), horizontalalignment='left',
#                     fontsize=8
#                    )
               
    ax.plot([0, 1], [0, 1], linestyle='--', label='"Always-Expansion Classifier')
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    ax.set_title('ROC Curve')
    plt.legend(frameon=True, loc='lower right')

def plot_heat_table(beta, annotation_df):

    import seaborn as sns

    beta_sort = np.sort(beta)[:, ::-1]

    to_plot = pd.concat([pd.Series(np.zeros(50)), 
                    pd.DataFrame(beta_sort[:, :10]),
                    pd.Series(np.zeros(50)), 
                    pd.Series(np.zeros(50))], 
                    axis=1, ignore_index=True)

    sns.set(font_scale=1.2)
    sns.set_style({"savefig.dpi": 100})
    # plot it out
    ax = sns.heatmap(to_plot, cmap=plt.cm.Blues, linewidths=.5,
                    annot=annotation_df, fmt = '')
    # set the x-axis labels on the top
    #ax.xaxis.tick_top()
    # rotate the x-axis labels
    #plt.xticks(rotation=90)
    # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)

    fig = ax.get_figure()
    fig.set_size_inches(20, 20)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig('heattable.png', format='png', dpi=600, transparent=True)

    plt.show()

