import unittest
from ignite.engine import Engine
from ignite.metrics import Loss, Accuracy
from metrics import CharacterErrorRate, WordErrorRate, Running
import torch

class DummyVocab:
    EOS = 'z'
    def char2int(self, c):
        return ord(c)
        
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
        self.cer = CharacterErrorRate()

    def test_cer_0(self):
        self.cer.update((['abc'], ['abc']))
        self.assertEqual(self.cer.compute(), 0)

    def test_cer_1(self):
        self.cer.update((['abc'], ['ab']))
        self.assertEqual(self.cer.compute(), 1/2)

    def test_cer_2(self):
        self.cer.update((['ab'], ['ade']))
        self.assertEqual(self.cer.compute(), 2/3)

    def test_cer_3(self):
        self.cer.update((['ab'], ['ade']))
        self.assertEqual(self.cer.compute(), 2/3)
        self.cer.update((['a'], ['a']))
        self.assertEqual(self.cer.compute(), (2/3 + 0/1)/2)
        self.cer.update((['de'], ['fgh']))
        self.assertEqual(self.cer.compute(), (2/3 + 0/1 + 3/3)/3)

    def tearDown(self):
        self.cer.reset()


class RunningCERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.dummy_engine = Engine(self.dummy_output)
        Running(CharacterErrorRate()).attach(self.dummy_engine, 'CER')

    def dummy_output(self, engine, batch):
        return batch

    def test_running_cer_0(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
            (['ab'], ['a']),
            (['dc'], ['dec']),
        ], 1, 3)
        self.assertEqual(state.metrics['CER'], (0/3+1/1+1/3)/3)

    def test_running_cer_1(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['CER'], (0/3))

    def test_running_cer_1(self):
        state = self.dummy_engine.run([
            (['ab'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['CER'], (1/3))

class WERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.wer = WordErrorRate()

    def test_wer_0(self):
        self.wer.update((['abc'], ['abc']))
        self.assertEqual(self.wer.compute(), 0)

    def test_wer_1(self):
        self.wer.update((['abc'], ['ab']))
        self.assertEqual(self.wer.compute(), 1)

    def test_wer_2(self):
        self.wer.update((['ab'], ['ade']))
        self.assertEqual(self.wer.compute(), 1)

    def test_wer_3(self):
        self.wer.update((['ab'], ['ade']))
        self.assertEqual(self.wer.compute(), 1)
        self.wer.update((['a'], ['a']))
        self.assertEqual(self.wer.compute(), (1 + 0)/2)
        self.wer.update((['de'], ['fgh']))
        self.assertEqual(self.wer.compute(), (1+0+1)/3)

    def tearDown(self):
        self.wer.reset()


class RunningWERTestCase(unittest.TestCase):
    def setUp(self):
        self.vocab = DummyVocab()
        self.dummy_engine = Engine(self.dummy_output)
        Running(WordErrorRate()).attach(self.dummy_engine, 'WER')

    def dummy_output(self, engine, batch):
        return batch

    def test_running_wer_0(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
            (['ab'], ['a']),
            (['dc'], ['dec']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], (0+1+1)/3)

    def test_running_wer_1(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], 0)

    def test_running_wer_1(self):
        state = self.dummy_engine.run([
            (['ab'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], 1)


if __name__ == '__main__':
    unittest.main()