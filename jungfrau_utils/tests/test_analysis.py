import unittest
import jungfrau_utils as ju
import numpy as np


class BaseTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_corrections(self):
        print(ju.corrections.test())
        self.assertTrue(ju.corrections.test())

    def test_pixel_map(self):
        data = np.random.randint(0, 60000, size=[1500, 1000], dtype=np.uint16)
        try:
            gain, data = ju.corrections.get_gain_data(data)
        except:
            self.assertTrue(False)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
