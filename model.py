import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


def generate_data(batch_size,input_length):
    dataset=torch.randint(1,10,[batch_size,input_length,2]).float().uniform_(0,1)
    return dataset

    # solution : (batch_size,input_length)
    # input_data : (batch_size,input_length,2)


def reward(solutions,input_data):
    batch_size = solutions.size(0)
    n = solutions.size(1)
    sample_solutions=torch.zeros([batch_size,n,2])
    for i in range(batch_size):
        sample_solutions[i]=input_data[i, solutions[i], :]
    actions=sample_solutions
    tour_len=torch.zeros([batch_size])
    for i in range(n-1):
        tour_len += torch.norm(actions[:, i+1, :]-actions[:, i, :], dim=1)
    tour_len += torch.norm(actions[:, 0, :]-actions[:, n-1, :], dim=1)
    return tour_len


class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,use_cuda):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_cuda=use_cuda
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)
        if use_cuda:
            self.h0 = self.h0.cuda()
            self.c0=self.c0.cuda()

    # embedding_inputs:(batch_size,input_length,embedding_dim)
    # hidden:# (n_layers,batch_size,hidden_dim)  注意这里hidden 包括h0与c0
    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """
        # (input_length,batch_size,embedding_dim)
        embedded_inputs = embedded_inputs.permute(1, 0, 2)    # 置换tensor的不同维度 但是不懂这里为什么要置换

        # outputs (input_length, batch_size, hidden_dim)
        # hidden (num_layers, batch_size, hidden_dim) hidden 包括hn,cn
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        # outputs (batch_size, input_length, hidden_dim)
        # hidden (num_layers, batch_size, hidden_dim) hidden 包括hn,cn
        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim,use_cuda=True):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        if use_cuda:
            V= V.cuda()
        self.V=V
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

    # input :  h_t : (batch_size,hidden_dim)
    # context:encoder_outputs: (batch_size, input_length, hidden_dim)
    # mask : (batch_size, input_length)
    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, input_length)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, input_length)
        context = context.permute(0, 2, 1)

        # (batch, hidden_dim, input_length)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, input_length)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # (batch, input_length)
        alpha = self.softmax(att)

        # (batch_size,hidden_dim)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        # (batch_size, input_length)
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    # embedding_inputs: (batch_size,input_length,embedding_dim)
    # decoder_inputs:# (batch_size,embedding_dim)
    # hidden: (2,batch_size,hidden_dim) 这里的2 一个是hn，一个是cn
    # context:encoder_outputs: (batch_size, input_length, hidden_dim)
    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch_size, input_length)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)

        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        # (input_length)
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i

        # (batch_size,input_length)
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # x : decoder_inputs: (batch_size,embedding_dim)
        # hidden: (2,batch_size,hidden_dim) 这里的2 一个是hn，一个是cn
        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            # h (batch_size,hidden_dim)
            # c (batch_size,hidden_dim)
            h, c = hidden

            # (batch_size,4*hidden_dim)
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            # (batch_size,hidden_dim)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            # c_t ：（batch_size,hidden_dim)
            c_t = (forget * c) + (input * cell)
            # h_t : (batch_size,hidden_dim)
            h_t = out * F.tanh(c_t)

            # Attention section
            # hidden_t: hidden_state:  (batch_size,hidden_dim)
            # output: alpha : (batch, input_length)
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))

            # (batch_size,hidden_dim)
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):

            # h_t : (batch_size,hidden_dim)
            # c_t : （batch_size,hidden_dim)
            # outs : alpha : (batch, input_length)
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            # (batch, input_length)
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            # max_probs : (batch_size)
            # indices : (batch_size)
            indices = masked_outs.multinomial(1).squeeze(1)
            # (batch_size,input_length)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()  #

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            # (batch_size,input_length,embedding_dim)
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            #
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        # outputs : (batch_size,t,input_length)
        # pointers : (batch_size,input_length)
        # hidden : （batch_size,hidden_dim)
        return (outputs, pointers), hidden



class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # (batch_size,embedding_dim)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # (batch_size*input_length,2)
        inputs = inputs.view(batch_size * input_length, -1)

        # (batch_size,input_length,embedding_dim)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        # (n_layers,batch_size,hidden_dim)  注意这里encoder_hidden0 包括h0与c0
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)

        # encoder_outputs (batch_size, input_length, hidden_dim)
        # encoder_hidden (num_layers, batch_size, hidden_dim) hidden 包括hn,cn
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)
        if self.bidir:
            decoder_hidden0 = (torch.cat(encoder_hidden[0][-2:], dim=-1),
                               torch.cat(encoder_hidden[1][-2:], dim=-1))
        else:
            # decoder_hidden0 是 encoder_hidden 对应 hn , cn最后一层的隐状态
            # decoder_hidden0 (layers,batch_size_hidden_dim) 这里的2 一个是hn，一个是cn
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])

        # outputs : (batch_size,t,input_length)
        # pointers : (batch_size,input_length)
        # decoder_hidden : （batch_size,hidden_dim)
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)
        # outputs : (batch_size,t,input_length)
        # pointers : (batch_size,input_length)
        return  outputs, pointers


class Attention_in_Critic(nn.Module):
    def __init__(self,hidden_dim):
        super(Attention_in_Critic,self).__init__()
        self.q_liner = nn.Linear(hidden_dim, hidden_dim)
        self.ref_liner = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self.V.data.uniform_(-1. / math.sqrt(hidden_dim), 1. / math.sqrt(hidden_dim))
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax()

    # encoder_outputs (batch_size, input_length, hidden_dim)
    # (batch_size,hidden_dim)
    def forward(self, encoder_output,encoder_hidden):
        encoder_output=encoder_output.permute(0,2,1)
        # (batch_size, hidden_dim,input_length)
        ref=self.ref_liner(encoder_output)
        # (batch_size,hidden_dim,input_length)
        q=self.q_liner(encoder_hidden).unsqueeze(2).expand(-1, -1, ref.size(2))
        # (batch_size, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(ref.size(0), -1).unsqueeze(1)
        # (batch_size,input_length)
        u=torch.bmm(V,self.tanh(ref+q)).squeeze(1)


        return ref,u


class Critic(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_blocks,
                 lstm_layers,
                 lstm_dropout,
                 bidir,
                 is_train,):
        super(Critic,self).__init__()
        self.is_train=is_train
        self.hidden_dim=hidden_dim
        self.n_precess_blocks=n_process_blocks
        self.embedding=nn.Linear(2,embedding_dim)
        self.encoder=Encoder(embedding_dim,hidden_dim,lstm_layers,lstm_dropout,bidir)
        self.decoder=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(hidden_dim),
            nn.Linear(hidden_dim,1)
        )
        self.att=Attention_in_Critic(hidden_dim)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_length = inputs.size(1)
        inputs=inputs.view(batch_size*input_length,-1)
        # (batch_size,input_length,embedding_dim)
        embedding_inputs=self.embedding(inputs).view(batch_size,input_length,-1)
        # (layers,batch_size,hidden_dim)
        encoder_hidden0=self.encoder.init_hidden(embedding_inputs)
        # encoder_outputs (batch_size, input_length, hidden_dim)
        # encoder_hidden (num_layers, batch_size, hidden_dim) hidden 包括hn,cn
        encoder_output, hidden=self.encoder(embedding_inputs,encoder_hidden0)
        # (batch_size,hidden_dim)
        process_block_state0 = hidden[0][-1]

        for _ in range(self.n_precess_blocks):
            # ref :  (batch_size, hidden_dim,input_length)
            # u :  (batch_size,input_length)
            ref,u=self.att(encoder_output,process_block_state0)
            # (batch_size,hidden_dim)
            process_block_state0=torch.bmm(ref,self.softmax(u).unsqueeze(2)).squeeze(2)

        out=self.decoder(process_block_state0).squeeze(1)
        return out


class NeuralCombOptRL(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_blocks,
                 objective_fn,  # reward function
                 lstm_layers,
                 dropout,
                 bidir,
                 is_train
                 ):
        super(NeuralCombOptRL,self).__init__()
        self.n_process_blocks=n_process_blocks
        self.objective_fn=objective_fn
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.is_train=is_train
        self.actor=PointerNet(embedding_dim,hidden_dim,lstm_layers,dropout,bidir)
        self.critic=Critic(embedding_dim,hidden_dim,n_process_blocks,lstm_layers,dropout,bidir,is_train)

    def forward(self, input_data):
        # outputs: (batch_size, t, input_length)
        # pointers : (batch_size,input_length)
        outputs, pointers = self.actor(input_data)
        batch_size=outputs.size(0)
        input_length=outputs.size(2)
        if self.is_train:
            probs = []
            for prob, action_id in zip(outputs, pointers):
                probs.append(prob[list(range(input_length)), action_id])
        else:

            probs = outputs

        V=self.critic(input_data)
        R=self.objective_fn(pointers,input_data)
        return R, V, probs,  pointers







