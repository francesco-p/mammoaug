import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def running_mean(x, N=16*10):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_loss_acc_per_batch(setting):

    AUG_df = pd.read_csv(f"/home/francesco/Documents/patch_titties/ablation/metrics_{setting}.csv")
    AUG_loss = AUG_df['loss'].values
    AUG_loss = running_mean(AUG_loss,32)
    AUG_acc = AUG_df['acc'].values
    AUG_acc = running_mean(AUG_acc,32)

    df = pd.read_csv(f"/home/francesco/Documents/patch_titties/ablation/NO_AUG_metrics_{setting}.csv")
    loss = df['loss'].values
    loss = running_mean(loss,32)
    acc = df['acc'].values
    acc = running_mean(acc,32)

    # plot loss
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    axs[0].plot(loss, color='red', linewidth=0.5, label='simple')
    axs[0].plot(AUG_loss, color='blue', linewidth=0.5, label='augmented')
    axs[0].set_title(f'{setting} - loss per batch (smoothed)')
    axs[0].set_xlabel("batch")
    axs[0].set_ylabel("loss")
    #axs[0].set_ylim(0,2)

    axs[1].plot(acc, color='red', linewidth=0.5, label='simple')
    axs[1].plot(AUG_acc, color='blue', linewidth=0.5, label='augmented')
    axs[1].set_title(f'{setting} - accuracy per batch (smoothed)')
    axs[1].set_xlabel("batch")
    axs[1].set_ylabel("acc")
    axs[1].set_ylim(0,1)

    axs[1].legend()
    axs[0].legend()
    axs[1].grid()
    axs[0].grid()

    plt.show(block=False)

def plot_loss_acc_epoch(setting):

    # Augmented    
    AUG_df = pd.read_csv(f"/home/francesco/Documents/patch_titties/ablation/metrics_{setting}.csv")
    AUG_df = AUG_df.groupby('epoch').mean()
    AUG_loss = AUG_df['loss'].values
    AUG_acc = AUG_df['acc'].values

    # Not augmented
    df = pd.read_csv(f"/home/francesco/Documents/patch_titties/ablation/NO_AUG_metrics_{setting}.csv")
    df = df.groupby('epoch').mean()
    loss = df['loss'].values
    acc = df['acc'].values

    # plot loss
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    axs[0].plot(loss, color='red', linewidth=2, label='simple')
    axs[0].plot(AUG_loss, color='blue', linewidth=2, label='augmented')
    axs[0].set_title(f'{setting} - loss per epoch')
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")

    axs[1].plot(acc, color='red', linewidth=2, label='simple')
    axs[1].plot(AUG_acc, color='blue', linewidth=2, label='augmented')
    axs[1].set_title(f'{setting} - accuracy per epoch')
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("acc")
    axs[1].set_ylim(0,1)
    
    axs[1].legend()
    axs[0].legend()
    axs[1].grid()
    axs[0].grid()
    plt.show(block=False)


plot_loss_acc_epoch('train')
plot_loss_acc_per_batch('train')
plot_loss_acc_epoch('val')
#plot_loss_acc_per_batch('val')

plt.show()