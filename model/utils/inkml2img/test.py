import inkml2img
import unittest


class Tests(unittest.TestCase):
    def test_output_path(self):
        input1 = "../data/101_fabrico.inkml"
        input2 = "404_blah.inkml"
        input3 = "data/blorg/242affa.inkml"
        self.assertEqual(
            inkml2img.get_output_path(input1, "train"), "train/101_fabrico.png"
        )
        self.assertEqual(
            inkml2img.get_output_path(input2, "train"), "train/404_blah.png"
        )
        self.assertEqual(
            inkml2img.get_output_path(input3, "train/test"), "train/test/242affa.png"
        )


if __name__ == "__main__":
    unittest.main()
