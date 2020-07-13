from dataset.collate import collate_images, collate_text
import torch
import unittest

class CollateTextTestCases(unittest.TestCase):
    def test_collate_text(self):
        a = [torch.tensor([1,2,3]), torch.tensor([1,2])]
        pad_val = 0
        collated_text, lengths = collate_text(a, pad_val)

        expected_text = torch.tensor([[1,2,3], [1,2,pad_val]])
        expected_lens = torch.tensor([3,2])
        self.assertTrue(collated_text.equal(expected_text))
        self.assertTrue(lengths.equal(expected_lens))
