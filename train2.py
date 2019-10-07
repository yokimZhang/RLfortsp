import torch
import os
import numpy as np
import torch.optim as optim
from model2 import NeuralCombOptRL
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from model2 import reward
import tsp
output_dir="model"
save_dir = os.path.join(os.getcwd(), output_dir)

# parameters
batch_size = 128
train_size = 100000
val_size = 1000
seq_len = 10

input_dim = 2
embedding_dim = 128
hidden_dim = 128
n_process_blocks = 3
n_glimpses = 1
use_tanh = True
C = 10   # tanh exploration
n_epochs = 1
use_cuda = True
random_seed = 111
is_train = True


actor_net_lr = 1e-4
critic_net_lr = 1e-4
actor_lr_decay_step = 5000
actor_lr_decay_rate = 0.96
critic_lr_decay_step = 5000
critic_lr_decay_rate = 0.96
log_step = 50


torch.manual_seed(random_seed)   # 初始化随机种子
reward_fn = reward

# instantiate the Neural Combinatorial Opt with RL module
model = NeuralCombOptRL(input_dim,
                        embedding_dim,
                        hidden_dim,
                        seq_len,
                        n_glimpses,
                        n_process_blocks,
                        C,
                        use_tanh,
                        reward_fn,    # 总距离
                        is_train,
                        use_cuda)

critic_mse = torch.nn.MSELoss()
critic_optim = optim.Adam(model.critic_net.parameters(), lr=critic_net_lr)
actor_optim = optim.Adam(model.actor_net.parameters(), lr=actor_net_lr)

actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
                                           list(range(actor_lr_decay_step,
                                                      actor_lr_decay_step * 1000,
                                                      actor_lr_decay_step)),
                                           gamma=actor_lr_decay_rate)

critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
                                            list(range(critic_lr_decay_step,
                                                       critic_lr_decay_step * 1000,
                                                       critic_lr_decay_step)),
                                            gamma=critic_lr_decay_rate)

if use_cuda:
    model = model.cuda()
    critic_mse = critic_mse.cuda()

training_dataset = tsp.TSPDataset(seq_len=seq_len, num_samples=train_size)
val_dataset = tsp.TSPDataset(num_samples=10)  # if specify filename, other arguments not required
training_dataloader = DataLoader(training_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=1)

validation_dataloader = DataLoader(val_dataset, batch_size=1,
                                   shuffle=True, num_workers=1)

step = 0
val_step = 0
epoch = 50

def train_one_epoch(i):
    global step
    # put in train mode!
    model.train()

    # sample_batch is [batch_size x sourceL x input_dim]
    for batch_id, sample_batch in enumerate(tqdm(training_dataloader, disable=False)):    # tqdm为进度条模块
        if use_cuda:
            sample_batch = sample_batch.cuda()

        R, v, probs, actions, actions_idxs = model(sample_batch)
        advantage = R - v  # means L(π|s)-b(s)

        # compute the sum of the log probs for each tour in the batch
        logprobs = sum([torch.log(prob) for prob in probs])
        # clamp any -inf's to 0 to throw away this tour
        logprobs[(logprobs < -1000).detach()] = 0.  # means log pθ(π|s)

        # multiply each time step by the advanrate
        reinforce = advantage * logprobs
        actor_loss = reinforce.mean()

        # actor net processing
        actor_optim.zero_grad()    # 在进行参数更新时要清零梯度，否则梯度会累积
        actor_loss.backward(retain_graph=True)   # 为什么要backward？？？
        # clip gradient norms
        torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(), max_norm=2.0, norm_type=2)   # 这一步又是在干啥？？？
        actor_optim.step()
        actor_scheduler.step()

        # critic net processing
        R = R.detach()
        critic_loss = critic_mse(v.squeeze(1), R)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(), max_norm=2.0, norm_type=2)
        critic_optim.step()
        critic_scheduler.step()

        step += 1

        # if not disable_tensorboard:
        #     writer.add_scalar('avg_reward', R.mean().item(), step)
        #     writer.add_scalar('actor_loss', actor_loss.item(), step)
        #     writer.add_scalar('critic_loss', critic_loss.item(), step)

        if step % log_step == 0:
            print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(i, batch_id, R.mean().item()))

def validation():
    global val_step
    model.actor_net.decoder.decode_type = 'beam_search'
    print('\n~Validating~\n')

    example_input = []
    example_output = []
    avg_reward = []

    # put in test mode!
    model.eval()

    for batch_id, val_batch in enumerate(tqdm(validation_dataloader, disable=False)):
        if use_cuda:
            val_batch = val_batch.cuda()

        R, probs, actions, action_idxs = model(val_batch)

        avg_reward.append(R[0].item())
        val_step += 1

        # if not disable_tensorboard:
        #     writer.add_scalar('val_avg_reward', R[0][0], int(val_step))

        if val_step % log_step == 0:
            print('Step: {}'.format(batch_id))

            # if plot_att:
            #     probs = torch.cat(probs, 0)
            #     plot_attention(example_input, example_output, probs.cpu().numpy())
    print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
    print('Validation overall reward var: {}'.format(np.var(avg_reward)))


def train_model():
    for i in range(epoch):
        if is_train:
            train_one_epoch(i)
        # Use beam search decoding for validation
        # validation()

        if is_train:
            model.actor_net.decoder.decode_type = 'stochastic'
            print('Saving model...epoch-{}.pt'.format(i))
            torch.save(model.state_dict(), os.path.join(save_dir, 'epoch-{}.pt'.format(i)))


if __name__ == '__main__':
    train_model()