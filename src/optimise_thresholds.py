import  numpy               as      np
import  matplotlib.pyplot   as      plt
from    sklearn             import  metrics

pred_file   = '/path/to/pred/file' 


def get_limits(vector):
    prev = vector[0]
    limits = []
    start = prev

    for l in vector[1:]:
        if l - prev > 1:
            limits.append(start)
            limits.append(prev)
            start = l

        prev = l
    limits.append(start)
    limits.append(prev)
    return limits

def main():
    n_thresh    = 1000

    pred_new    = np.loadtxt(pred_file, delimiter=' ', dtype=str)

    y_pred_new  = pred_new[:,1].astype(float)

    y_true_new  = np.array([int(f.split('_')[0]) for f in pred_new[:,0]])

    thresholds  = np.linspace(0, 1, n_thresh+1)

    accs_new    = np.zeros_like(thresholds)
    f1_new      = np.zeros_like(thresholds)

    for _i, t in enumerate(thresholds, 0):
        an      = metrics.accuracy_score(y_true_new, y_pred_new > t)
        fi      = metrics.f1_score(y_true_new, y_pred_new > t)

        accs_new[_i] = an
        f1_new[_i]   = fi

    plt.figure(figsize=(15, 8))
    
    anew,       = plt.plot(accs_new, 'b-')

    maxacc_new  = accs_new.max()

    fnew,       = plt.plot(f1_new, 'b-.')

    plt.legend(
            [
                anew,
                fnew,
            ],
            [
                'Accuracy. Best threshold = %.4f' %(thresholds[accs_new.argmax()]),
                'F1 score. Best threshold = %.4f' %(thresholds[f1_new.argmax()]),
            ]
    )

    plt.grid()

    xtick_locs, _ = plt.xticks()
    ticks       = range(0,n_thresh+1,100)
    plt.xticks(ticks, ['%.2f' %(thresholds[_xt]) for _xt in ticks])

    for mw in get_limits(np.where(accs_new == maxacc_new)[0]):
        plt.text(mw, maxacc_new, '%.4f' %(thresholds[mw]), rotation=30, verticalalignment='bottom')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Accuracies on val set plotted by threshold.')
    plt.tight_layout()

    plt.savefig('thresholds.png')
    plt.show()

if __name__ == '__main__':
    main()
