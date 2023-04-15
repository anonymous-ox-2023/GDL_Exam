import datetime

from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import models as models
from helper import cal_loss
from my_utils import *
from utils import progress_bar, save_model


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()

    for batch_idx, data in enumerate(trainloader):
        data, label = data['pointcloud'].to(device).float(), data['category'].to(device)
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            data, label = data['pointcloud'].to(device).float(), data['category'].to(device)
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':

    # =============== Parameters to change ======================================================
    batch_size = 8
    workers = 4

    path = Path(r'Path to the raw folder of ModelNet40 dataset')

    # Select training mode: original, gat, edge, dynamic
    mode = "gat"

    # ===========================================================================================

    train_transforms = transforms.Compose([
        PointSampler(5000),
        HiddenPointRemover(),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])

    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

    train_loader = DataLoader(dataset=train_ds, batch_size=8, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=8)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)

    net = models.__dict__['pointMLP'](num_classes=5, mode=mode)
    checkpoint_path = 'checkpoints/best_checkpoint.pth'

    net = net.to(device)

    print("loaded model")

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, 300, eta_min=0.005, last_epoch=start_epoch - 1)

    criterion = cal_loss

    for epoch in range(start_epoch, 15):
        train_out = train(net, train_loader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
        print("trained")
        test_out = validate(net, valid_loader, criterion, device)
        print("validated")
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        print("best_train_acc: ", best_train_acc)
        print("best_test_acc: ", best_test_acc)

        print("best_train_loss: ", best_train_loss)
        print("best_test_loss: ", best_test_loss)

        save_model(
            net, epoch, path="checkpoint", acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
