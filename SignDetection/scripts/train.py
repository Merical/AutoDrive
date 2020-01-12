import torch.utils.data as Data
from SignDetection.scripts.SignDetectionPack.models import *
from SignDetection.scripts.SignDetectionPack.tools import *
import time
import os
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

EPOCH = 300
BATCH_SIZE = 32
LR = 0.001
CLASS_NUM = 6
with_gpu = True
datadir = "../data/"
weightdir = "../weights"
writer = SummaryWriter("../logs")

traindir = os.path.join(datadir, "train")
valdir = os.path.join(datadir, "val")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
)


val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        # transforms.Resize((299,299)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=with_gpu
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=with_gpu
)

# model = get_inception_v3(CLASS_NUM)
model = get_shufflenet_v2_0_5(CLASS_NUM)
writer.add_graph(model, torch.rand(1, 3, 224, 224))

if with_gpu:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()
best_prec1 = torch.FloatTensor([0])


tic = time.time()
for epoch in range(EPOCH):
    adjust_learning_rate(optimizer, epoch, LR)
    loss_train, top1_train = train(train_loader, model, loss_func, optimizer, epoch, with_gpu)
    loss_valid, top1_valid = validate(val_loader, model, loss_func, with_gpu)
    writer.add_scalar("Train/Top1", top1_valid, epoch)
    writer.add_scalar("Train/Loss", loss_valid, epoch)
    writer.add_scalar("Validation/Top1", top1_valid, epoch)
    writer.add_scalar("Validation/Loss", loss_valid, epoch)
    writer.flush()

    prec1 = top1_valid

    if with_gpu:
        prec1 = prec1.cpu()

    is_best = bool(prec1.numpy() >= best_prec1.numpy())
    # Get greater Tensor
    best_prec1 = torch.FloatTensor(max(prec1.numpy(), best_prec1.numpy()))

    if is_best:
        torch.save(model.state_dict(), 'best_weights.pth')
    torch.save(model.state_dict(), 'checkpoint.pth')
writer.close()

