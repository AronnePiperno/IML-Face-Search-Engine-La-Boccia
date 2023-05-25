from datetime import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class adjust_lr:
    """
    create a class instance;
    multiply DECAY every STEP epochs
    """
    def __init__(self, step, decay):
        self.step = step
        self.decay = decay

    def adjust(self, optimizer, epoch, lr):
        lr = lr * (self.decay ** (epoch // self.step))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr

def train(epochs, net, device, trainloader, testloader, classes, scheduler, lr, inner_param, sigma, hlen, eta, save):

    print('==> Preparing training data..')

    criterion = DualClasswiseLoss(num_classes=classes, inner_param=inner_param, sigma=sigma, feat_dim=hlen, use_gpu=True)

    best_epoch = 0
    best_loss = 1e4

    optimizer = optim.SGD([
            {'params': net.parameters(), 'weight_decay': 5e-4},
            {'params':  criterion.parameters(), 'weight_decay': 5e-4}
        ], lr=lr, momentum=0.9)

    since = time.time()
    for epoch in range(epochs):
        print('==> Epoch: %d' % (epoch + 1))
        net.train()
        dcdh_loss = AverageMeter()
        scheduler.adjust(optimizer, epoch)
        # epoch_start = time.time()
        for batch_id, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            hash_bits = net(imgs)
            loss_dual = criterion(hash_bits, labels)
            hash_binary = torch.sign(hash_bits)
            batchY = EncodingOnehot(labels, classes).cuda()
            W = torch.pinverse(batchY.t() @ batchY) @ batchY.t() @ hash_binary           # Update W

            batchB = torch.sign(torch.mm(batchY, W) + eta * hash_bits)  # Update B

            loss_vertex = (hash_bits - batchB).pow(2).sum() / len(imgs)     # quantization loss
            loss_h = loss_dual + eta * loss_vertex

            dcdh_loss.update(loss_h.item(), len(imgs))
            loss_h.backward()
            optimizer.step()

        print("[epoch: %d]\t[hashing loss: %.3f ]" % (epoch+1, dcdh_loss.avg))

        if (epoch+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                # centers_trained = torch.sign(criterion.centers.data).cuda()
                trainB, train_labels = compute_result(trainloader, net, device)
                testB, test_labels = compute_result(testloader, net, device)
                mAP = compute_mAP(trainB, testB, train_labels, test_labels, device)
                print('[Evaluate Phase] Epoch: %d\t mAP: %.2f%%' % (epoch+1, 100. * float(mAP)))

        if dcdh_loss.avg < best_loss:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save({'backbone': net.state_dict(),
                        'centers': criterion.state_dict()}, './checkpoint/%s' % save)
            best_loss = dcdh_loss.avg
            best_epoch = epoch

        if (epoch - best_epoch) > epochs // 4:
            print("Training terminated at epoch %d" %(epoch + 1))
            break

    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s with best loss in epoch {}".format(time_elapsed // 60, time_elapsed % 60, best_epoch + 1))
    print("Model saved as %s" % save)


class DualClasswiseLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, inner_param, sigma=0.25, use_gpu=True):
        super(DualClasswiseLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.sigma = sigma
        self.inner_param = inner_param
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: shape of (batch_size, feat_dim).
            labels: shape of (batch_size, ) or (batch_size, 1)
        """

        #   compute L_1 with single constraint.
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        dist_div = torch.exp(-0.5*self.sigma*distmat)/(torch.exp(-0.5*self.sigma*distmat).sum(dim=1, keepdim=True) + 1e-6)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.view(-1, 1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist_log = torch.log(dist_div+1e-6) * mask.float()
        loss = -dist_log.sum() / batch_size

        #   compute L_2 with inner constraint on class centers.
        centers_norm = F.normalize(self.centers, dim=1)
        theta_x = 0.5 * self.feat_dim * centers_norm.mm(centers_norm.t())
        mask = torch.eye(self.num_classes, self.num_classes).bool().cuda()
        theta_x.masked_fill_(mask, 0)
        loss_iner = Log(theta_x).sum() / (self.num_classes*(self.num_classes-1))

        loss_full = loss + self.inner_param * loss_iner
        return loss_full

def Log(x):
    """
    Log trick for numerical stability
    """

    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.tensor([0.]).cuda())

    return lt
class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()# first reset

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def EncodingOnehot(target, nclasses):
    target_onehot = torch.Tensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.cpu().view(-1, 1), 1)
    return target_onehot

def compute_result(dataloader, net, device):

    """
    return hashing codes of data with shape (N, len_bits) and its labels (N, )
    """
    hash_codes = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs, cls = imgs.to(device), cls.to(device)
        hash_values = net(imgs)
        hash_codes.append(hash_values.data)
        label.append(cls)

    hash_codes = torch.cat(hash_codes)
    B = torch.where(hash_codes > 0.0, torch.tensor([1.0]).cuda(), torch.tensor([-1.0]).cuda())

    return B, torch.cat(label)
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device):

    AP = []
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        N = torch.sum(correct)
        Ns = torch.arange(1, N+1).float().to(device)
        index = (torch.nonzero(correct, as_tuple=False)+1)[:, 0].float()
        AP.append(torch.mean(Ns / index))

    mAP = torch.mean(torch.Tensor(AP))
    return mAP

if __name__ == "__main__":
    pass
