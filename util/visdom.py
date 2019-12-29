import numpy as np

import visdom


class VisdomLogger():
    def __init__(self, port, env_name):
        self.vis = visdom.Visdom(env = env_name, port=port)

    def numpify(self, tensor):
        return tensor.cpu().detach().numpy()
    
    def visualize_masks(self, imgs, masks, recons):
        # print('recons min/max', recons.min().item(), recons.max().item())
        recons = np.clip(recons, 0., 1.)
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
        colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors[1:]])
        colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors[1:]])
        colors.extend([(c[0]//8, c[1]//8, c[2]//8) for c in colors[1:]])
    
        masks = np.argmax(masks, 1)
        seg_maps = np.zeros_like(imgs)
        for i in range(imgs.shape[0]):
            for y in range(imgs.shape[2]):
                for x in range(imgs.shape[3]):
                    seg_maps[i, :, y, x] = colors[masks[i, y, x]]
        seg_maps /= 255.0
        self.vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])
         
    
    def log_visdom(self, images, masks, recons, num_imgs=8):
        self.visualize_masks(self.numpify(images[:num_imgs]),
                             self.numpify(masks[:num_imgs]),
                             self.numpify(recons[:num_imgs]))
