#
# This file is part of "Scale Space for Morning Project Samurai (SS for MPS)"
# SS for MPS is used for lectures by Morning Project Samurai (MPS).
#
# SS for MPS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SS for MPS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SS for MPS.  If not, see <http://www.gnu.org/licenses/>.
#
# (c) Junya Kaneko <jyuneko@hotmail.com>
#

import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'


class GaussianFilter:
    """
    ガウシアンフィルタ。

    例:
    gaussian_filter = GaussianFilter(1.6)
    gaussian_filter(image)
    """
    def __init__(self, sigma):
        self._sigma = sigma
        self._filter = self._generate_filter()
        logging.debug('Created filter for %s' % sigma)

    @property
    def sigma(self):
        return self._sigma

    def _generate_filter(self):
        """
        ガウシアンカーネルの -6\sigma から +6\sigma までの値を用いたガウシアンフィルタを返す。
        :return: ガウシアンフィルタ
        """
        kernel = np.array([0.0, ] * int(2 * np.ceil(6 * self._sigma) + 1))
        for i, x in enumerate(range(-int(len(kernel)/2) + 1, int(len(kernel)/2))):
            kernel[i] = (1/(self._sigma * np.sqrt(2 * np.pi))) * np.exp(-np.power(x, 2)/(2 * np.power(self._sigma, 2)))
        return kernel / kernel.sum()

    def apply(self, gray_image):
        """
        ガウシアンフィルタを gray_image に適用した結果を返す。

        二次元のガウシアンカーネルを用いたコンボリューションは、一次元のガウシアンカーネルを用いたコンボリューションを
        x 軸方向と y 軸方向に順に適応したものと同じ。

        I(x, y) * G(x, y, sigma) = I(x, y) * Gx(x, y, sigma) * Gy(x, y, sigma)

        ここで、* はコンボリューションを表す。

        :param gray_image: ガウシアンフィルタを適用するグレースケールイメージ
        :return: ガウシアンフィルタ適用後の gray_image
        """
        img1 = Image(self.sigma, width=gray_image.width, height=gray_image.height)
        for i in range(len(gray_image)):
            for j in range(len(gray_image[i])):
                for k, val in enumerate(self._filter):
                    h_index = j + k - self.center_index
                    if h_index < 0:
                        img1[i, j] += gray_image[i, 0] * val
                    elif h_index >= len(gray_image[i]):
                        img1[i, j] += gray_image[i, -1] * val
                    else:
                        img1[i, j] += gray_image[i, h_index] * val
        img2 = Image(img1.sigma, width=img1.width, height=img1.height)
        for i in range(len(img1)):
            for j in range(len(img1[i])):
                for k, val in enumerate(self._filter):
                    v_index = i + k - self.center_index
                    if v_index < 0:
                        img2[i, j] += img1[i, j] * val
                    elif v_index >= len(img1):
                        img2[i, j] += img1[-1, j] * val
                    else:
                        img2[i, j] += img1[v_index, j] * val
        return Image(self._sigma, img2)

    @property
    def center_index(self):
        return int(len(self)/2)

    def __len__(self):
        return len(self._filter)

    def __call__(self, gray_img):
        return self.apply(gray_img)


class FilterDict(dict):
    """
    作成したフィルタを保持しておく。

    注意: __iter__ が返すイテレータは list のイテレータなので、イテレート中に FilterDict の内容が更新されても、
    その更新はイテレータに反映されない。
    """
    def __init__(self, filters=()):
        super().__init__({})
        for _filter in filters:
            self[_filter.sigma] = _filter

    def add(self, _filter):
        self[_filter.sigma] = _filter

    def __iter__(self):
        return sorted(self.keys()).__iter__()


class Image(np.ndarray):
    """
    スケールスペースで使用するグレースケール画像。

    スケールスペース中で扱いやすいように numpy.ndarray のサブクラスとして作成。
    """
    def __new__(cls, sigma, np_array=None, width=0, height=0):
        """
        :param sigma: スケールスペースにおける画像の σ の値
        :param np_array: グレースケール画像。すでにあるグレースケール画像を用いてインスタンスを作る場合に指定する。
        :param width: 画像の幅。新たに画像を作成する場合に指定する。
        :param height: 画像の高さ。新たに画像を作成する場合に指定する。
        :return:
        """
        if np_array is not None:
            obj = np.asarray(np_array).view(cls)
        else:
            width = width if isinstance(width, int) else int(width)
            height = height if isinstance(height, int) else int(height)
            obj = np.asarray(np.array([[0] * width] * height)).view(cls)
        obj._sigma = sigma
        logging.debug('Created the image for %s (%s x %s)' % (sigma, len(obj[0]), len(obj)))
        return obj

    @property
    def sigma(self):
        return self._sigma

    @property
    def width(self):
        return len(self[0])

    @property
    def height(self):
        return len(self)

    def has_larger_sigma(self, image):
        return self.sigma > image.sigma

    def has_smaller_sigma(self, image):
        return self.sigma < image.sigma

    def has_same_sigma(self, image):
        return self.sigma == image.sigma


class DoGImage(np.ndarray):
    """
    Difference of Gaussian によって作成された画像。

    スケールスペース中の画像として扱いやすいように numpy.ndarray のサブクラスとして作成。
    """
    def __new__(cls, image0, image1):
        """
        image0 と image1 はともに元となる画像。画像の順番は気にしなくて良い。

        σの値の大きいものから小さいものが自動的に引かれる。

        :param image0: 元となる画像1
        :param image1: 元となる画像2
        :return:
        """
        source_images = [image0, image1] if image0.has_smaller_sigma(image1) else [image1, image0]
        obj = np.asarray(source_images[1] - source_images[0]).view(cls)
        obj._source_images = source_images
        logging.debug('Created the DoG image for %s:%s' % (source_images[0].sigma, source_images[1].sigma))
        return obj

    @property
    def sources(self):
        """
        :return: 2枚の元画像から成るタプル。(σ の小さい画像, σ の大きい画像)。

        """
        return tuple(self._source_images)


class ScaleSpace:
    """
    Scale Space

    インスタンス化されると同時に Scale Space と それに基づく DoG の空間を生成して保持する。
    また、next_octave プロパティを使うことによって、次のオクターブの空間を生成し、それを返す。

    Scale space を構成する画像は、images プロパティを使って参照可能。並び順は σ の小さいもの順。
    また、イテレータを用いて for image in scale_space のように参照することもできる。

    Dog の空間は、DoG_space プロパティを用いて参照可能。並び順は元画像と同様。
    """
    def __init__(self, gray_image, sigma, s, filters=FilterDict()):
        if isinstance(gray_image, np.ndarray) and not isinstance(gray_image, Image):
            gray_image = Image(0, gray_image)
        elif not isinstance(gray_image, Image):
            raise ValueError('gray_img must be either numpy.ndarray type or scale_space.Image type.')

        self._images = [gray_image, ]
        self._sigma = sigma
        self._s = s

        self._update_filter_dict(filters)
        self._filters = filters

        self._generate_space()
        self._dog_space = self._generate_dog_space()
        self._next_octave = None

    @property
    def _k(self):
        return np.power(2, 1 / self._s)

    @property
    def images(self):
        return tuple(self._images)

    @property
    def DoG_space(self):
        return self._dog_space

    @property
    def next_octave(self):
        if not self._next_octave:
            self._next_octave = self._generate_next_octave()
        return self._next_octave

    def _update_filter_dict(self, filter_set):
        for i in range(0, self._s + 1):
            sigma = np.power(self._k, i) * self._sigma
            if sigma not in filter_set:
                filter_set.add(GaussianFilter(sigma))

    def _generate_space(self):
        for sigma in self._filters:
            self._images.append(self._filters[sigma].apply(self._images[0]))

    def _generate_dog_space(self):
        dogs = []
        logging.debug('Start to create DoGs.')
        for i in range(1, len(self._images)):
            dogs.append(DoGImage(self._images[i], self._images[i - 1]))
        return dogs

    def _generate_next_octave(self):
        source_image = self._images[-1]
        sampled_image = Image(0, width=source_image.width/2, height=source_image.height/2)

        for i in range(0, len(source_image), 2):
            for j in range(0, len(source_image[i]), 2):
                sampled_image[i / 2][j / 2] = source_image[i][j]
        return ScaleSpace(sampled_image, self._sigma, self._s, self._filters)

    def __iter__(self):
        return self._images.__iter__()


if __name__ == '__main__':
    import cv2
    import os

    # 元となるイメージの読み込み
    orig_img = cv2.imread('imgs/lena.jpg')
    gray_img = Image(0, cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY))

    # 最初のスケールスペースを作成
    scale_space = ScaleSpace(gray_img, 1.6, 2)
    logging.debug('Created the first octave.')

    # 2オクターブまで作成
    current_octave = scale_space
    for i in range(0, 3):
        # 現在のスケールスペースを構成する画像を保存
        base_dir_path = 'imgs/scale_space/octave%s' % i
        for image in current_octave:
            if not os.path.isdir(base_dir_path):
                os.mkdir(base_dir_path)
            cv2.imwrite(os.path.join(base_dir_path, '%s.jpg' % image.sigma), image)
            logging.debug('Wrote image: %s.' % os.path.join(base_dir_path, '%s.jpg' % image.sigma))

        # 現在のスケールスペースを構成する画像に DoG を適用した画像を保存
        for image in current_octave.DoG_space:
            dir_path = os.path.join(base_dir_path, 'dog_space/')
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            cv2.imwrite(os.path.join(dir_path, '%s-%s.jpg' % (image.sources[0].sigma, image.sources[1].sigma)), image)
            logging.debug('Wrote image: %s.' % os.path.join(dir_path, '%s-%s.jpg' % (image.sources[0].sigma, image.sources[1].sigma)))

        # 次のオクターブへ
        current_octave = current_octave.next_octave
