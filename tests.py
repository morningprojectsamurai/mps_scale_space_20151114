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


import unittest
import cv2
from scale_space import GaussianFilter


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'


class GaussianFilterTestCase(unittest.TestCase):
    def test_filter_generation(self):
        gaussian_filter = GaussianFilter(1.0)
        self.assertEqual(len(gaussian_filter), 13)
        self.assertEqual(gaussian_filter.center_index, 6)
        self.assertEqual(gaussian_filter._filter.sum(), 1)

    def test_apply(self):
        orig_img = cv2.imread('imgs/lena.jpg')
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        gaussian_filter = GaussianFilter(1.0)
        filtered_img = gaussian_filter(gray_img)

        self.assertTrue(len(gray_img) == len(filtered_img) and len(gray_img[0]) == len(filtered_img[0]))

        for i in range(len(filtered_img)):
            for j in range(len(filtered_img[i])):
                self.assertTrue(filtered_img[i, j] < 256)

if __name__ == '__main__':
    unittest.main()
