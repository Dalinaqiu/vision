import cv2
import numpy as np
import os

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1 / 255.0 if val_scale == 1 else 1

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255)
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val = mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple, list)):
            self.mean_val = [int(x * factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):
    def __init__(self, size):
        assert type(size) in [int, tuple], "CHECK SIZE TYPE!"   # 断言确认size类型为int或tuple，int型指定输出图像为正方形，tuple类型可以指定输出图像为任意矩形
        if isinstance(size, int):
            self.size = (size, size)    # int需转化为tuple，保持后面的处理代码统一
        else:
            self.size = size

    def __call__(self, image, label):
        h, w = image.shape[:2]
        center_h, center_w = int(h/2), int(w/2)

        try:
            h_start, w_start = center_h - int(self.size[0]/2), center_w - int(self.size[1]/2)
            h_end, w_end = h_start + self.size[0], w_start + self.size[1]

            image = image[h_start:h_end, w_start:w_end, :]
            label = label[h_start:h_end, w_start:w_end]
        except Exception as e:
            print('CROP OUT OF IMAGE, RETURN ORIGIN IMAGE!')
        return image, label


# TODO
class Resize(object):
    def __init__(self, size):
        assert type(size) in [int, tuple], "CHECK SIZE TYPE!"   # 参考朱老师代码即可
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, label=None):
        image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return image, label


# TODO
class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob    # 水平翻转概率为prob，竖直翻转概率为1-prob

    def __call__(self, image, label=None):
        if np.random.rand() <= self.prob:
            image = image[:, ::-1, :]   # 逆序遍历w即水平翻转
            if label is not None:
                label = label[:, ::-1]
        else:
            image = image[::-1, :, :]   # 逆序遍历h即竖直翻转
            if label is not None:
                label = label[::-1, :]
        return image, label


# TODO
class RandomCrop(object):
    def __init__(self, size):
        assert type(size) in [int, tuple], "CHECK SIZE TYPE!"
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, label=None):
        h, w = image.shape[:2]

        try:
            h_start = np.random.randint(0, h - self.size[0] + 1)
            w_start = np.random.randint(0, w - self.size[1] + 1)
            h_end, w_end = h_start + self.size[0], w_start + self.size[1]

            image = image[h_start:h_end, w_start:w_end, :]
            if label is not None:
                label = label[h_start:h_end, w_start:w_end]
        except Exception as e:
            print('CROP OUT OF IMAGE, RETURN ORIGIN IMAGE!')
        return image, label


# TODO
class Scale(object):
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, image, label=None):
        image = cv2.resize(image, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        return image, label


# TODO
class RandomScale(object):
    def __init__(self, scales=[0.5, 1, 2]):
        self.scales = scales

    def __call__(self, image, label=None):
        scale = float(np.random.choice(self.scales))    # 随机挑选一种scale，也可以按步长取scale
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, label


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))    # 获取当前文件目录的绝对路径
    image_path = os.path.join(current_path, "dummy_data/JPEGImages/2008_000064.jpg")    # 搞不清楚当前工作目录的相对路径的同学，建议写文件地址的时候统一为绝对路径
    label_path = os.path.join(current_path, "dummy_data/GroundTruth_trainval_png/2008_000064.png")

    # image = cv2.imread('./dummy_data/JPEGImages/2008_000064.jpg')
    # label = cv2.imread('./dummy_data/GroundTruth_trainval_png/2008_000064.png')
    image = cv2.imread(image_path, 1)
    label = cv2.imread(label_path, 0)
    print("origin size:", image.shape, label.shape)

    # TODO: crop_size
    crop_size = 500     # 涉及crop时，指定最终输出图像大小
    pad_size = 512      # pad_size一般来说应大于等于crop_size，防止crop时越界
    # TODO: Transform: RandomScale, RandomFlip, Pad, RandomCrop
    transform = Compose([RandomScale(scales=[0.5, 1, 2]),
                         RandomFlip(prob=0.5),
                         Pad(size=pad_size),
                         RandomCrop(size=crop_size)])   # 组合transform方法

    # 单一transform方法
    # transform = RandomScale(scales=[0.5,1,2])   # 此方法输出size应为原图的0.5，1,  2倍随机大小
    # transform = RandomFlip(prob=0.5)    # 此方法输出size与原图一致，上下翻转概率为0.5，左右翻转概率为0.5
    # transform = Pad(size=pad_size)    # 当pad_size大于原图任意一边长，该边长变为pad_size，如取512时，图像表现为黑边
    # transform = CenterCrop(size=crop_size)    # 输出10张一致的从中间裁剪的大小为crop_size的图像，注意crop_size应小于min(h,w), 否则操作越界报错
    # transform = RandomCrop(size=crop_size)    #  输出10张随机裁剪的大小为crop_size的图像

    for i in range(10):
        # TODO: call transform
        transformed_image, transformed_label = transform(image, label)
        print("transformed size:", transformed_image.shape, transformed_label.shape)
        # TODO: save image
        save_path = os.path.join(current_path, 'save')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path,'{}.jpg'.format(i)), transformed_image)
        cv2.imwrite(os.path.join(save_path,'{}.png'.format(i)), transformed_label)
        print("save result to {}".format(os.path.join(save_path,'{}.jpg'.format(i))))


if __name__ == "__main__":
    main()
