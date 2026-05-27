import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def safe_state(silent):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    # Use buffer_rgba for compatibility with newer matplotlib versions
    data = np.asarray(fig.canvas.buffer_rgba())
    data = data[..., :3]  # drop alpha channel
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img
