import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import *

batchs=1000
batch_size=128
input_length=10
embedding_dim=128
n_process_blocks=1
hidden_dim=512
nof_lstms=1
dropout=0
actor_net_lr = 1e-4
critic_net_lr = 1e-4
actor_lr_decay_step = 5000
actor_lr_decay_rate = 0.96
critic_lr_decay_step = 5000
critic_lr_decay_rate = 0.96
log_step = 10
bidir=False
is_train=True
random_seed=111

torch.manual_seed(random_seed)   # 初始化随机种子

model = NeuralCombOptRL(embedding_dim,
                        hidden_dim,
                        n_process_blocks,
                        reward,
                        nof_lstms,
                        dropout,
                        bidir,
                        is_train
                       )

critic_mse = torch.nn.MSELoss()
critic_optim = optim.Adam(model.critic.parameters(), lr=critic_net_lr)
actor_optim = optim.Adam(model.actor.parameters(), lr=actor_net_lr)

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

step = 0
val_step = 0
epoch = 50
data=[]
for i in range(batchs):
    data.append(generate_data(batch_size,input_length))

def train_one_epoch(i):
    global step
    # put in train mode!
    model.train()

    # sample_batch is [batch_size x sourceL x input_dim]
    batch_id=0
    for  sample_batch in data:
        batch_id+=1
        # if use_cuda:
        #     sample_batch = sample_batch.cuda()

        # prob : (batch_size,input_length)
        R, V, probs, actions_idxs = model(sample_batch)
        advantage = R - V  # means L(π|s)-b(s)
        # compute the sum of the log probs for each tour in the batch
        logprobs = [sum(torch.log(prob)) for prob in probs]
        logprobs=torch.tensor(logprobs)
        # clamp any -inf's to 0 to throw away this tour
        logprobs[(logprobs < -1000).detach()] = 0.  # means log pθ(π|s)

        # multiply each time step by the advanrate
        reinforce = advantage * logprobs
        actor_loss = reinforce.mean()

        # actor net processing
        actor_optim.zero_grad()    # 在进行参数更新时要清零梯度，否则梯度会累积
        actor_loss.backward(retain_graph=True)   # 为什么要backward？？？
        # clip gradient norms
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=2.0, norm_type=2)   # 这一步又是在干啥？？？
        actor_optim.step()
        actor_scheduler.step()

        # critic net processing
        R = R.detach()
        critic_loss = critic_mse(V, R)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_norm=2.0, norm_type=2)
        critic_optim.step()
        critic_scheduler.step()

        step += 1

        # if not disable_tensorboard:
        #     writer.add_scalar('avg_reward', R.mean().item(), step)
        #     writer.add_scalar('actor_loss', actor_loss.item(), step)
        #     writer.add_scalar('critic_loss', critic_loss.item(), step)

        if step % log_step == 0:
            print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(i, batch_id, R.mean().item()))


def train_model():

    for i in range(epoch):
        train_one_epoch(i)
        # Use beam search decoding for validation




if __name__ == '__main__':
    train_model()



# if torch.cuda.is_available():
#     USE_CUDA = True
#     print('Using GPU, %i devices.' % torch.cuda.device_count())
# else:
#     USE_CUDA = False

# input_data=generate_data(batch_size,input_length)
# Model = PointerNet(embedding_size,
#                    hiddens,
#                    nof_lstms,
#                    dropout,
#                    bidir)
#
#
# o,p=Model(input_data)
# tour_len=reward(p,input_data)
# print(input_data)
# print(p)
# print(tour_len)

# Model=Critic(embedding_size,
#              hiddens,
#              n_process_blocks,
#              nof_lstms,
#              dropout,
#              bidir)
# out=Model(input_data)
# print(input_data)
# print(out)








