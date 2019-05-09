import numpy as np
import matplotlib.pyplot as plt
# load the history file
path_out = './outdir/'
npzfile = np.load(path_out + 'history.npz')
# summarize history for accuracy
plt.plot(npzfile['acc'])
plt.plot(npzfile['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(path_out + 'accuracies.png', bbox_inches='tight')
plt.clf()
# summarize history for loss
plt.plot(npzfile['loss'])
plt.plot(npzfile['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(path_out + 'losses.png', bbox_inches='tight')
plt.clf()