import unittest
from ignite.engine import Engine
from ignite.metrics import Loss, Accuracy
from metrics import CharacterErrorRate, WordErrorRate, Running
import torch
from torch.utils.data import DataLoader, TensorDataset

class DummyVocab:
    EOS = 'z'
    def char2int(self, c):
        return ord(self.EOS)
        
def str2tensor(text):
    if not isinstance(text, list):
        text = [text]
    max_length = max([len(t) for t in text])
    text = [list(t) + [DummyVocab.EOS]*(max_length + 1 - len(t)) for t in text]
    text = torch.LongTensor([list(map(ord, t)) for t in text])
    return text

class CERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.cer = CharacterErrorRate(self.vocab, batch_first=True)

    def test_str2tensor(self):
        tensor = str2tensor(['ab', 'cd'])
        self.assertTrue(torch.all(tensor == (torch.tensor([[97, 98, 122], [99, 100, 122]]))))

    def test_cer_0(self):
        self.cer.update((str2tensor('abc'), str2tensor('abc')))
        self.assertEqual(self.cer.compute(), torch.tensor(0))

    def test_cer_1(self):
        self.cer.update((str2tensor('abc'), str2tensor('ab')))
        self.assertEqual(self.cer.compute(), torch.tensor(1/2))

    def test_cer_2(self):
        self.cer.update((str2tensor('ab'), str2tensor('ade')))
        self.assertEqual(self.cer.compute(), torch.tensor(2/3))

    def test_cer_3(self):
        self.cer.update((str2tensor('ab'), str2tensor('ade')))
        self.assertEqual(self.cer.compute(), torch.tensor(2/3))
        self.cer.update((str2tensor('a'), str2tensor('a')))
        self.assertEqual(self.cer.compute(), torch.tensor((2/3 + 0/1)/2))
        self.cer.update((str2tensor('de'), str2tensor('fgh')))
        self.assertEqual(self.cer.compute(), torch.tensor((2/3 + 0/1 + 3/3)/3))

    def tearDown(self):
        self.cer.reset()


class RunningCERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.dummy_engine = Engine(self.dummy_output)
        Running(CharacterErrorRate(self.vocab, batch_first=True)).attach(self.dummy_engine, 'CER')

    def dummy_output(self, engine, batch):
        return batch

    def test_str2tensor(self):
        tensor = str2tensor(['ab', 'cd'])
        self.assertTrue(torch.all(tensor == (torch.tensor([[97, 98, 122], [99, 100, 122]]))))

    def test_running_cer_0(self):
        state = self.dummy_engine.run([
            (str2tensor('abc'), str2tensor('abc')),
            (str2tensor('ab'), str2tensor('a')),
            (str2tensor('dc'), str2tensor('dec')),
        ], 1, 3)
        self.assertEqual(state.metrics['CER'], torch.tensor((0/3+1/1+1/3)/3))

    def test_running_cer_1(self):
        state = self.dummy_engine.run([
            (str2tensor('abc'), str2tensor('abc')),
        ], 1, 3)
        self.assertEqual(state.metrics['CER'], torch.tensor((0/3)))

    def test_running_cer_1(self):
        state = self.dummy_engine.run([
            (str2tensor('ab'), str2tensor('abc')),
        ], 1, 3)
        self.assertEqual(state.metrics['CER'], torch.tensor((1/3)))

class WERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.wer = WordErrorRate(self.vocab, batch_first=True)

    def test_str2tensor(self):
        tensor = str2tensor(['ab', 'cd'])
        self.assertTrue(torch.all(tensor == (torch.tensor([[97, 98, 122], [99, 100, 122]]))))

    def test_wer_0(self):
        self.wer.update((str2tensor('abc'), str2tensor('abc')))
        self.assertEqual(self.wer.compute(), torch.tensor(0))

    def test_wer_1(self):
        self.wer.update((str2tensor('abc'), str2tensor('ab')))
        self.assertEqual(self.wer.compute(), torch.tensor(1))

    def test_wer_2(self):
        self.wer.update((str2tensor('ab'), str2tensor('ade')))
        self.assertEqual(self.wer.compute(), torch.tensor(1))

    def test_wer_3(self):
        self.wer.update((str2tensor('ab'), str2tensor('ade')))
        self.assertEqual(self.wer.compute(), torch.tensor(1))
        self.wer.update((str2tensor('a'), str2tensor('a')))
        self.assertEqual(self.wer.compute(), torch.tensor((1 + 0)/2))
        self.wer.update((str2tensor('de'), str2tensor('fgh')))
        self.assertEqual(self.wer.compute(), torch.tensor((1+0+1)/3))

    def tearDown(self):
        self.wer.reset()


class RunningWERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.dummy_engine = Engine(self.dummy_output)
        Running(WordErrorRate(self.vocab, batch_first=True)).attach(self.dummy_engine, 'WER')

    def dummy_output(self, engine, batch):
        return batch

    def test_str2tensor(self):
        tensor = str2tensor(['ab', 'cd'])
        self.assertTrue(torch.all(tensor == (torch.tensor([[97, 98, 122], [99, 100, 122]]))))

    def test_running_wer_0(self):
        state = self.dummy_engine.run([
            (str2tensor('abc'), str2tensor('abc')),
            (str2tensor('ab'), str2tensor('a')),
            (str2tensor('dc'), str2tensor('dec')),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], torch.tensor((0+1+1)/3))

    def test_running_wer_1(self):
        state = self.dummy_engine.run([
            (str2tensor('abc'), str2tensor('abc')),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], torch.tensor(0))

    def test_running_wer_1(self):
        state = self.dummy_engine.run([
            (str2tensor('ab'), str2tensor('abc')),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], torch.tensor(1))


if __name__ == '__main__':
    unittest.main()