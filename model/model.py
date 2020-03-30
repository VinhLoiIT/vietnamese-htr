import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import get_attention

__all__ = [
    'Model', 'ModelTF', 'ModelRNN'
]

class Model(nn.Module):
    def __init__(self, cnn, vocab):
        super().__init__()
        self.cnn = cnn
        self.vocab = vocab

    def embed_image(self, images):
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
        #image_features = self.pe_image(image_features) # [B,C',H',W']
        image_features = image_features.transpose(-2, -1) # [B,C',W',H']
        image_features = image_features.reshape(batch_size, self.cnn.n_features, -1) # [B, C', S=W'xH']
        image_features = image_features.transpose(1,2) # [B, S, C']
        return image_features

    def embed_text(self, text):
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

    def forward(self, images, labels):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
            - labels: [B,T]
            - image_sizes: [B,2]
            - label_lengths: [B]

        Returns:
            - outputs: [B,T,V]
        '''
        images = self.embed_image(images) # [B,S,C']
        labels = self.embed_text(labels) # [B,T,V]
        outputs = self._forward_decode(images, labels) # [B,T,V]
        return outputs

    def _forward_decode(self, embed_image, embed_text):
        '''
        Shapes:
        -------
            - embed_image: [B,S,E]
            - embed_text: [B,T,E]

        Returns:
            - outputs: [B,T,V]
        '''
        pass

    def greedy(self, images, start, max_length):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
            - start: [B]
            - max_length: int
        Returns:
            - outputs: [B,T]
        '''
        pass

class ModelTF(Model):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab)
        self.Ic = nn.Linear(cnn.n_features, config['attn_size'])
        self.Vc = nn.Linear(self.vocab.size, config['attn_size'])
        self.character_distribution = nn.Linear(config['attn_size'], self.vocab.size)

        if config['use_encoder']:
            self.transformer = nn.Transformer(
                    d_model=config['attn_size'],
                    nhead=config['nhead'],
                    num_encoder_layers=config['encoder_nlayers'],
                    num_decoder_layers=config['decoder_nlayers'],
                    dim_feedforward=config['dim_feedforward'],
                    dropout=config['dropout'],
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(config['attn_size'], config['nhead'],
                                                       dim_feedforward=config['dim_feedforward'],
                                                       dropout=config['dropout'])
            self.transformer = nn.TransformerDecoder(decoder_layer, config['decoder_nlayers'])

        #self.pe_text = PositionalEncoding1d(config['attn_size'], batch_first=True)
        #self.pe_image = PositionalEncoding2d(self.cnn.n_features)

    def embed_image(self, image):
        image = super().embed_image(image) # [B,S,C']
        image = self.Ic(image) # [B,S,E]
        return image

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
        # output = self.transformer(embed_image, embed_text, tgt_mask=attn_mask)
        outputs = self.transformer(embed_text, embed_image, tgt_mask=attn_mask)
        outputs.transpose_(0,1)
        outputs = self.character_distribution(outputs)
        return outputs

    def greedy(self, images, start, max_length):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
            - start: [B]
            - max_length: int
        Returns:
            - outputs: [B,T]
        '''
        batch_size = len(images)
        images = self.embed_image(images) # [B,S,E]
        images.transpose_(0, 1) # [S,B,E]

        predicts = start.unsqueeze_(-1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, max_length).to(predicts.device)
        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for t in range(max_length):
            targets = self.embed_text(predicts) # [B,T,E]
            targets = targets.transpose_(0,1) # [T,B,E]
            # output = self.transformer(images, targets, tgt_mask=attn_mask[:t+1, :t+1]) # [T,B,E]
            output = self.transformer(targets, images, tgt_mask=attn_mask[:t+1, :t+1]) # [T,B,E]
            output = output.transpose_(0, 1) # [B,T,E]
            output = self.character_distribution(output[:,[-1]]) # [B,1,V]
            output = output.argmax(-1) # [B, 1]
            predicts = torch.cat([predicts, output], dim=1)

            end_flag |= (output.cpu().squeeze(-1) == self.vocab.char2int(self.vocab.EOS))
            if end_flag.all():
                break
        return predicts[:,1:]

class ModelRNN(Model):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab)
        self.hidden_size = config.hidden_size
        attn_size = config.attn_size

        self.rnn = nn.LSTMCell(
            input_size=self.vocab.size+attn_size,
            hidden_size=self.hidden_size,
        )

        self.Ic = nn.Linear(cnn.n_features, attn_size)
        self.Hc = nn.Linear(self.hidden_size, attn_size)
        self.attention = get_attention(config.attention, attn_size)
        self.teacher_forcing_ratio = config.teacher_forcing_ratio

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

    def greedy(self, images, start, max_length):
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
        rnn_input = self.embed_text(start.unsqueeze(-1)).squeeze_(1).float() # [B,V]

        hidden = self._init_hidden(batch_size).to(embedded_image.device) # [B, H]
        cell_state = self._init_hidden(batch_size).to(embedded_image.device) # [B, H]

        outputs = torch.zeros(batch_size, max_length, device=embedded_image.device, dtype=torch.long)

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for t in range(max_length):
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
