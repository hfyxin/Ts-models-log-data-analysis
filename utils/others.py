import matplotlib.pyplot as plt

def plot_and_save(x, y, fn):
    '''plot and save to an image, used for loss curve.'''
    if x == None:
        plt.plot(y)
    else:
        plt.plot(x, y)
    plt.savefig(fn)


def print_train_info(epoch, time, loss, metrics):
    print('epoch {} - {:.0f}s. Train Loss: {:.4f}, Acc: {:.2%}, P: {:.2%}, R: {:.2%}'.format(
        epoch, time, loss.result(),
        metrics['accuracy'].result(),
        metrics['precision'].result(),
        metrics['recall'].result()))


def print_train_info_v2(epoch, time, history):
    print('epoch {} - {:.0f}s.'.format(epoch, time), end=' ')
    for key, value in history.items():
        if 'acc' in key:
            print('{}: {:.2%}'.format(key, value[-1]), end=' ')
        else:
            print('{}: {:.4f}'.format(key, value[-1]), end=' ')
    print('')