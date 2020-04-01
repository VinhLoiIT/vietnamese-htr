import torch
from model.positional_encoding import PositionalEncoding1d, PositionalEncoding2d
import unittest
import math

torch.manual_seed(0)

class PE1DTestCase(unittest.TestCase):

    def setUp(self):
        d_model = 4
        l = 10
        self.pe = PositionalEncoding1d(d_model, batch_first=True, dropout=0, max_len=l)

    def test_1_sample_1_step(self):
        x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320]]]) # [1,1,4]
        x2 = self.pe(x)

        expected_x = x + torch.tensor([[[0, 1, 0, 1]]])
        self.assertTrue(x2.equal(expected_x))

    def test_1_sample_2_step(self):
        x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                           [0.3074, 0.6341, 0.4901, 0.8964]]])
        x2 = self.pe(x)

        expected_x = x + torch.tensor([[[0, 1, 0, 1],
                                        [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]]])
        self.assertTrue(expected_x.allclose(x2))

    def test_2_sample_1_step(self):
        x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320]],
                          [[0.3074, 0.6341, 0.4901, 0.8964]]]) # [2,1,4]
        x2 = self.pe(x)

        expected_x = x + torch.tensor([[[0, 1, 0, 1]],
                                       [[0, 1, 0, 1]]])
        self.assertTrue(x2.equal(expected_x))

    def test_2_sample_2_step(self):
        x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                           [0.3074, 0.6341, 0.4901, 0.8964]],

                          [[0.4556, 0.6323, 0.3489, 0.4017],
                           [0.0223, 0.1689, 0.2939, 0.5185]]]) # [2,2,4]

        x2 = self.pe(x)

        expected_x = x + torch.tensor([[[0, 1, 0, 1],
                                        [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]],
                                        
                                       [[0, 1, 0, 1],
                                        [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]]])
        self.assertTrue(expected_x.allclose(x2))


class PE2DTestCase(unittest.TestCase):

    def setUp(self):
        d_model = 4
        l = 10
        self.pe = PositionalEncoding2d(d_model, dropout=0, max_len=l)

    def test_convenient(self):
        x1 = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320]]]]) # [B,H,W,C]=[1,1,1,4]
        x2 = torch.tensor([[[[0.4963]], [[0.7682]], [[0.0885]], [[0.1320]]]]) # [B,C,H,W]=[1,4,1,1]
        self.assertTrue(x2.equal(x1.permute(0,3,1,2)))

    def test_1_sample_1x1(self):
        x = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320]]]]) # [1,1,1,4]
        x2 = self.pe(x.permute(0,3,1,2))

        expected_x = x + torch.tensor([[[[0, 1, 0, 1]]]])
        self.assertTrue(x2.equal(expected_x.permute(0,3,1,2)))

    def test_1_sample_2x1(self):
        x = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320]],
                           [[0.3074, 0.6341, 0.4901, 0.8964]]]]) # [1,2,1,4]
        x2 = self.pe(x.permute(0,3,1,2))

        expected_x = x + torch.tensor([[[[0, 1, 0, 1]],
                                        [[0, 1, 0, 1]]]])
        self.assertTrue(x2.equal(expected_x.permute(0,3,1,2)))

    def test_1_sample_1x2(self):
        x = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320],
                            [0.3074, 0.6341, 0.4901, 0.8964]]]]) # [1,1,2,4]
        x2 = self.pe(x.permute(0,3,1,2))

        expected_x = x + torch.tensor([[[[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]]]])
        self.assertTrue(x2.allclose(expected_x.permute(0,3,1,2)))

    def test_1_sample_2x2(self):
        x = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320],
                            [0.3074, 0.6341, 0.4901, 0.8964]],

                           [[0.4556, 0.6323, 0.3489, 0.4017],
                            [0.0223, 0.1689, 0.2939, 0.5185]]]]) # [1,2,2,4]
        x2 = self.pe(x.permute(0,3,1,2))

        expected_x = x + torch.tensor([[[[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]],
                                        [[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]]]])
        self.assertTrue(x2.allclose(expected_x.permute(0,3,1,2)))

    def test_2_sample_2x2(self):
        x = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320],
                            [0.3074, 0.6341, 0.4901, 0.8964]],

                           [[0.4556, 0.6323, 0.3489, 0.4017],
                            [0.0223, 0.1689, 0.2939, 0.5185]]],

                          [[[0.6977, 0.8000, 0.1610, 0.2823],
                            [0.6816, 0.9152, 0.3971, 0.8742]],

                           [[0.4194, 0.5529, 0.9527, 0.0362],
                            [0.1852, 0.3734, 0.3051, 0.9320]]]]) # [2,2,2,4]

        x2 = self.pe(x.permute(0,3,1,2))

        expected_x = x + torch.tensor([[[[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]],
                                        [[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]]],

                                       [[[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]],
                                        [[0, 1, 0, 1],
                                         [math.sin(1), math.cos(1), math.sin(1/100.), math.cos(1/100.)]]]])
        self.assertTrue(x2.allclose(expected_x.permute(0,3,1,2)))