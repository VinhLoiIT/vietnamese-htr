import unittest
from ignite.engine import Engine
from ignite.metrics import Loss, Accuracy
from metrics import CharacterErrorRate, WordErrorRate, Running
import torch

class CERTestCase(unittest.TestCase):
    def setUp(self):
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

class WERWordTestCase(unittest.TestCase):
    def setUp(self):
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

    def test_wer_4(self):
        self.wer.update((['word0 word1 word2'], ['word1 word2 word3']))
        self.assertEqual(self.wer.compute(), 2/3)

    def test_wer_5(self):
        self.wer.update((['word1'], ['word0']))
        self.assertEqual(self.wer.compute(), 1/1)

    def test_wer_6(self):
        self.wer.update((['word1 word2 word3'], ['word1 word2 word3']))
        self.assertEqual(self.wer.compute(), 0/3)

    def test_wer_7(self):
        self.wer.update((['word2 word3 '], ['word2 word1']))
        self.assertEqual(self.wer.compute(), 2/2)

    def tearDown(self):
        self.wer.reset()


class RunningWERTestCase(unittest.TestCase):
    def setUp(self):
        self.dummy_engine = Engine(self.dummy_output)
        Running(WordErrorRate()).attach(self.dummy_engine, 'WER')

    def dummy_output(self, engine, batch):
        return batch

    def test_running_wer_0(self):
        state = self.dummy_engine.run([
            (['word1 word2 word3'], ['word0 word1 word2']),
            (['word1 word2 word3'], ['word1 word2 word3']),
            (['word1'], ['word0']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], (2/3 + 0/3 + 1/1)/3)

    def test_running_wer_1(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], 0)

    def test_running_wer_2(self):
        state = self.dummy_engine.run([
            (['ab'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], 1)

    def test_running_wer_3(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
            (['ab'], ['a']),
            (['dc'], ['dec']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], (0+1+1)/3)

    def test_running_wer_4(self):
        state = self.dummy_engine.run([
            (['abc'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], 0)

    def test_running_wer_5(self):
        state = self.dummy_engine.run([
            (['ab'], ['abc']),
        ], 1, 3)
        self.assertEqual(state.metrics['WER'], 1)


if __name__ == '__main__':
    unittest.main()