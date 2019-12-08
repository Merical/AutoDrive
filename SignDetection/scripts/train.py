import torch.utils.data as Data
from SignDetection.scripts.SignDetectionPack.models import *
import time

EPOCH = 1
BATCH_SIZE = 4
LR = 0.001
CLASS_NUM = 5
with_gpu = False

# train_dataset = torch.load('./train_dataset_transfer.pt')
# train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# test_dataset = torch.load('./valid_dataset_transfer.pt')
# x_test, y_test = test_dataset.tensors
# x_test = x_test.cuda() if with_gpu else x_test
# y_test = y_test.cuda() if with_gpu else y_test
# x_valid = x_test[:BATCH_SIZE*5]
# y_valid = y_test[:BATCH_SIZE*5]

model = get_inception_v3(CLASS_NUM)

if with_gpu:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()
best_accuracy = 0
best_loss = 1

tic = time.time()
for epoch in range(EPOCH):
    if epoch % 20 == 0:
        LR = LR * 0.9
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)

    for step, (x, y) in enumerate(train_loader):
        b_x = x.cuda() if with_gpu else x
        b_y = y.cuda() if with_gpu else y

        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            for i in range(int(x_valid.size(0))//BATCH_SIZE):
                if i == 0:
                    valid_output = model(x_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                else:
                    valid_output = torch.cat((valid_output, model(x_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE])), dim=0)
            pred_y = torch.max(valid_output, 1)[1].cuda().data if with_gpu else torch.max(valid_output, 1)[1].data
            accuracy = torch.sum(pred_y == y_valid).type(torch.FloatTensor) / y_valid.size(0)
            if best_accuracy <= accuracy and best_loss >= loss:
                torch.save(model.state_dict(), 'cvd_best_weights.pth')
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| valid accuracy: %.4f' % accuracy)
toc = time.time()
print('the training proceed took ', toc - tic, ' seconds.')

del y_valid, x_valid, train_loader, train_dataset, y_test, x_test
# del model, y_valid, x_valid, train_loader, train_dataset, y_test, x_test
if EPOCH:
    del loss_func, optimizer, output, pred_y, valid_output, b_x, b_y, x, y

# time.sleep(2)

test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# model = CVD_model()
# model.load_state_dict(torch.load('cvd_best_weights.pth'))
# model = model.cuda() if with_gpu else model
# model.eval()

accuracy = 0.
for step, (x, y) in enumerate(test_loader):
    b_x = x.cuda() if with_gpu else x
    b_y = y.cuda() if with_gpu else y
    output = model(b_x)
    pred_y = torch.max(output, 1)[1].cuda().data if with_gpu else torch.max(output, 1)[1].data
    accuracy += torch.sum(pred_y == b_y).type(torch.FloatTensor)
accuracy /= test_dataset.tensors[0].size(0)
accuracy = float(accuracy)
print('real accuracy: ', float(accuracy))

# for i in range(int(x_test.size()[0])//BATCH_SIZE):
#     if i == 0:
#         test_output = model(x_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
#     else:
#         test_output = torch.cat((test_output, model(x_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE])), dim=0)
# pred_y = torch.max(test_output, 1)[1].cuda().data if with_gpu else torch.max(test_output, 1)[1].data
#
# print(pred_y, 'prediction number')
# print(y_test, 'real number')
# print('real accuracy: ', float(torch.sum(pred_y == y_test).type(torch.FloatTensor) / y_test.size(0)))