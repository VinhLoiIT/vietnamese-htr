import torch
import torch.nn as nn
import torch.nn.functional as F

from queue import PriorityQueue
from typing import List, Dict, Tuple
from .attention import get_attention
from .positional_encoding import PositionalEncoding1d, PositionalEncoding2d

__all__ = [
    'Model', 'ModelTF', 'ModelRNN', 'ModelTFEncoder',
]

class _BeamSearchNode(object):
    def __init__(self,
        prev_chars: List,
        prev_node: '_BeamSearchNode',
        current_char: int,
        log_prob: float,
        length: int
    ):
        self.prev_chars = prev_chars
        self.prev_node = prev_node
        self.current_char = current_char
        self.log_prob = log_prob
        self.length = length

    def eval(self):
        return self.log_prob / float(self.length - 1 + 1e-6)

    def __lt__(self, other):
        return self.length < other.length

    def new(self, char_index: int, log_prob: float):
        new_node = _BeamSearchNode(
            self.prev_chars + [self.current_char],
            self,
            char_index,
            self.log_prob + log_prob,
            self.length + 1,
        )
        return new_node

class Model(nn.Module):
    '''
    Base class for all models
    '''
    def __init__(self, cnn, vocab):
        super().__init__()
        self.cnn = cnn
        self.vocab = vocab

    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]

        Returns:
        --------
            - image_features: [B,S,C']
        '''
        image_features = self.cnn(images) # [B, C', H', W']
        batch_size, height, width = images.size(0), images.size(2), images.size(3)
        image_features = image_features.transpose(-2, -1) # [B,C',W',H']
        image_features = image_features.reshape(batch_size, self.cnn.n_features, -1) # [B, C', S=W'xH']
        image_features = image_features.transpose(1,2) # [B, S, C']
        return image_features

    def embed_text(self, text: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - text: [B,T]

        Returns:
        --------
            - text: [B,T,V]
        '''
        text = F.one_hot(text, self.vocab.size).float().to(text.device)
        return text

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
            - labels: [B,T]

        Returns:
            - outputs: [B,T,V]
        '''
        images = self.embed_image(images) # [B,S,C']
        labels = self.embed_text(labels) # [B,T,V]
        outputs = self._forward_decode(images, labels) # [B,T,V]
        return outputs

    def _forward_decode(self, embed_image: torch.Tensor, embed_text: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - embed_image: [B,S,E]
            - embed_text: [B,T,E]

        Returns:
            - outputs: [B,T,V]
        '''
        pass

    def greedy(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
        Returns:
            - outputs: [B,T]
        '''
        pass

    def beamsearch(self, images: torch.Tensor, beam_width: int) -> torch.Tensor:
        pass

class ModelTF(Model):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab)
        self.register_buffer('start_index', torch.tensor(vocab.SOS_IDX, dtype=torch.long))
        self.max_length = config['max_length']

        self.Ic = nn.Linear(cnn.n_features, config['attn_size'])
        self.Vc = nn.Linear(vocab.size, config['attn_size'])
        self.character_distribution = nn.Linear(config['attn_size'], vocab.size)

        decoder_layer = nn.TransformerDecoderLayer(config['attn_size'], config['nhead'],
                                                    dim_feedforward=config['dim_feedforward'],
                                                    dropout=config['dropout'])
        self.decoder = nn.TransformerDecoder(decoder_layer, config['decoder_nlayers'])

        if config.get('use_pe_text', False):
            self.pe_text = PositionalEncoding1d(config['attn_size'], batch_first=True)
        else:
            self.pe_text = lambda x: x

        if config.get('use_pe_image', False):
            self.pe_image = PositionalEncoding2d(config['attn_size'])
        else:
            self.pe_image = lambda x: x

    def embed_image(self, images):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]

        Returns:
        --------
            - image_features: [B,S,E]
        '''
        image_features = self.cnn(images) # [B, C', H', W']
        image_features = image_features.permute(0,2,3,1) # [B, H', W', C']
        image_features = self.Ic(image_features) # [B,H',W',E]
        image_features = image_features.permute(0,3,1,2) # [B, E, H', W']
        image_features = self.pe_image(image_features) # [B,E,H',W']
        batch_size, height, width = images.size(0), images.size(2), images.size(3)
        image_features = image_features.transpose(-2, -1) # [B,E,W',H']
        image_features = image_features.reshape(batch_size, self.Ic.out_features, -1) # [B, E, S=W'xH']
        image_features = image_features.transpose(1,2) # [B, S, E]
        return image_features

    def embed_text(self, text):
        '''
        Shapes:
        -------
            - text: [B,T]

        Returns:
        --------
            - text: [B,T,A]
        '''
        text = super().embed_text(text) # [B,T,V]
        text = self.Vc(text) # [B,T,E]
        text = self.pe_text(text)
        return text

    def _forward_decode(self, embed_image, embed_text):
        '''
        Shapes:
        -------
            - embed_image: [B,S,E]
            - embed_text: [B,T,E]

        Returns:
        --------
            - outputs: [B,T,V]
        '''
        embed_image.transpose_(0, 1) # [S,B,E]
        embed_text.transpose_(0, 1) # [T,B,E]

        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, embed_text.size(0)).to(embed_text.device)
        outputs = self.decoder(embed_text, embed_image, tgt_mask=attn_mask)
        outputs.transpose_(0,1)
        outputs = self.character_distribution(outputs)
        return outputs

    def greedy(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
        Returns:
            - outputs: [B,T]
        '''
        batch_size = len(images)
        images = self.embed_image(images) # [B,S,E]
        images.transpose_(0, 1) # [S,B,E]

        predicts = self.start_index.expand(batch_size).unsqueeze(-1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, self.max_length).to(predicts.device)
        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for t in range(self.max_length):
            targets = self.embed_text(predicts) # [B,T,E]
            targets = targets.transpose_(0,1) # [T,B,E]
            output = self.decoder(targets, images, tgt_mask=attn_mask[:t+1, :t+1]) # [T,B,E]
            output = output.transpose_(0, 1) # [B,T,E]
            output = self.character_distribution(output[:,[-1]]) # [B,1,V]
            output = output.argmax(-1) # [B, 1]
            predicts = torch.cat([predicts, output], dim=1)

            end_flag |= (output.cpu().squeeze(-1) == self.vocab.char2int(self.vocab.EOS))
            if end_flag.all():
                break
        return predicts[:,1:]

    def inference_step(self, embedded_image: torch.Tensor, predicts: torch.Tensor):
        '''
        Shapes:
        -------
            - embedded_image: [B,S,E]
            - predicts: [B,T]
            - output: [B,V]
        '''
        text = self.embed_text(predicts) # [B,T,E]
        text = text.transpose_(0,1) # [T,B,E]
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, len(text)).to(text.device)
        embedded_image = embedded_image.transpose(0, 1) # [S,B,E]
        output = self.decoder(text, embedded_image, tgt_mask=attn_mask) # [T,B,E]
        output = output.transpose_(0, 1) # [B,T,E]
        output = self.character_distribution(output[:,[-1]]) # [B,1,V]
        output = F.log_softmax(output.squeeze(1), -1) # [B,V]
        return output


    def beamsearch(self, images: torch.Tensor, start: torch.Tensor, max_length: int, beam_width: int):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
            - start: [B]
        Returns:
        --------
            - outputs: [B,T]
            - lengths: [B]
        '''

        def decode_one_sample(image, start, max_length, beam_width):
            '''
            image: [S,E]
            start: [1]
            '''
            node = _BeamSearchNode([], None, start.item(), 0, 1)
            nodes = []
            endnodes = []

            # start the queue
            nodes = PriorityQueue()
            nodes.put_nowait((-node.eval(), node))

            # start beam search
            image = image.unsqueeze(0) # [B=1,S,E]
            while True:
                # give up when decoding takes too long
                if nodes.qsize() > 2000: break

                # fetch the best node
                score: float
                node: _BeamSearchNode
                score, node = nodes.get()
                if node.current_char == self.vocab.char2int(self.vocab.EOS) and node.prev_node is not None:
                    endnodes.append((score, node))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= beam_width:
                        break
                    else:
                        continue

                # decode for one step using decoder
                predicts = torch.tensor(node.prev_chars + [node.current_char], dtype=torch.long).unsqueeze_(0).to(image.device) # [B=1,T]
                output = self.inference_step(image, predicts) # [B=1, V]

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_probs, indexes = output.topk(beam_width, -1)
                for log_prob, index in zip(log_probs.squeeze_(0).tolist(), indexes.squeeze_(0).tolist()):
                    new_node = node.new(index, log_prob)
                    nodes.put_nowait((-new_node.eval(), new_node))

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get_nowait() for _ in range(beam_width)]

            # Only get the maximum prob
            # print([x[0] for x in endnodes])
            score, node = sorted(endnodes, key=lambda x: x[0])[0]
            # print(f'Best score: {-score}, length = {node.length}')
            s = [node.current_char]
            # back trace
            while node.prev_node is not None:
                node = node.prev_node
                s.insert(0, node.current_char)
            return torch.tensor(s, dtype=torch.long)

        batch_size = len(images)
        images = self.embed_image(images) # [B,S,E]
        images.transpose_(0, 1) # [S,B,E]

        predicts = start.unsqueeze_(-1)
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            string_index = decode_one_sample(images[:, idx], start[idx], max_length, beam_width)
            decoded_batch.append(string_index)

        return torch.nn.utils.rnn.pad_sequence(decoded_batch, batch_first=True)[:, 1:]

class ModelTFEncoder(ModelTF):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab, config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['attn_size'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config['encoder_nlayers'])

    def embed_image(self, images):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]

        Returns:
        --------
            - image_features: [B,S,E]
        '''
        images = super().embed_image(images) # B,S,E
        images = images.transpose(0, 1) # S,B,E
        images = self.encoder(images) # S,B,E
        images = images.transpose(0, 1) # B,S,E
        return images

class ModelRNN(Model):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab)
        self.hidden_size = config['hidden_size']
        self.register_buffer('start_index', torch.tensor(vocab.SOS_IDX, dtype=torch.long))
        self.max_length = config['max_length']
        attn_size = config['attn_size']

        self.rnn = nn.LSTMCell(
            input_size=self.vocab.size+attn_size,
            hidden_size=self.hidden_size,
        )

        self.Ic = nn.Linear(cnn.n_features, attn_size)
        self.Hc = nn.Linear(self.hidden_size, attn_size)
        self.attention = get_attention(config['attention'], attn_size)
        self.teacher_forcing_ratio = config['teacher_forcing_ratio']

        self.character_distribution = nn.Linear(self.hidden_size, self.vocab.size)

    def _init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def _forward_decode(self, embed_image, embed_text):
        '''
        Shapes:
        -------
            - embed_image: tensor of [B, S, C]
            - embed_text: tensor of [B, T, V], each target has <start> at beginning of the word

        Returns:
        --------
            - outputs: tensor of [B, T, V]
        '''
        batch_size = embed_image.size(0)
        max_length = embed_text.size(1)
        embed_image = self.Ic(embed_image) # [B, S, E]

        rnn_input = embed_text[:, 0].float() # [B,V]
        hidden = self._init_hidden(batch_size).to(embed_image.device) # [B,H]
        cell_state = self._init_hidden(batch_size).to(embed_image.device) # [B,H]

        outputs = torch.zeros(batch_size, max_length, self.vocab.size, device=embed_image.device)
        for t in range(max_length):
            attn_hidden = self.Hc(hidden) # [B, E]
            context, _ = self.attention(attn_hidden.unsqueeze(1), embed_image, embed_image) # [B, 1, attn_size], [B, 1, S]
            context = context.squeeze_(1) # [B, attn_size]
            # self.rnn.flatten_parameters()
            hidden, cell_state = self.rnn(torch.cat((rnn_input, context), -1), (hidden, cell_state))
            output = self.character_distribution(hidden) # [B, V]
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            if self.training and teacher_force:
                rnn_input = embed_text[:, t]
            else:
                output = output.argmax(-1)
                rnn_input = F.one_hot(output, self.vocab.size).float().to(outputs)

        return outputs

    def greedy(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
            - start: [B]
            - max_length: int
        Returns:
            - outputs: [B,T]
        '''
        embedded_image = self.embed_image(images) # [B,S,C']

        num_pixels = embedded_image.size(1)
        batch_size = embedded_image.size(0)

        embedded_image = self.Ic(embedded_image)
        rnn_input = self.start_index.expand(batch_size).float() # [B,V]

        hidden = self._init_hidden(batch_size).to(embedded_image.device) # [B, H]
        cell_state = self._init_hidden(batch_size).to(embedded_image.device) # [B, H]

        outputs = torch.zeros(batch_size, self.max_length, device=embedded_image.device, dtype=torch.long)

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for t in range(self.max_length):
            attn_hidden = self.Hc(hidden) # [B, A]
            context, _ = self.attention(attn_hidden.unsqueeze(1), embedded_image, embedded_image) # [B, 1, A]
            context.squeeze_(1) #
            rnn_input = torch.cat((rnn_input, context), -1) # [B, V+A]

            hidden, cell_state = self.rnn(rnn_input, (hidden, cell_state))
            output = self.character_distribution(hidden) # [B,V]
            output = output.argmax(-1)
            outputs[:, t] = output
            rnn_input = F.one_hot(output, self.vocab.size).float().to(outputs.device)

            end_flag |= (output.cpu().squeeze(-1) == self.vocab.char2int(self.vocab.EOS))
            if end_flag.all():
                break
        return outputs
