import datetime

from torch.backends import cudnn
from torch.utils.data import DataLoader

import models as models
from helper import cal_loss
from my_utils import *

if __name__ == '__main__':

    # =============== Parameters to change ======================================================
    batch_size = 8
    workers = 4

    path = Path(r'Path to the raw folder of ModelNet40 dataset')

    # Select training mode: original, gat, edge, dynamic
    mode = "gat"

    # ===========================================================================================

    train_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])

    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)  # True
    valid_loader = DataLoader(dataset=valid_ds, batch_size=1)

    # valid_loader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=4,
    #                                                   batch_size=8, shuffle=False, drop_last=False)

    print("loaded")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)

    net = models.__dict__['pointMLP']()
    checkpoint_path = 'checkpoints/best_checkpoint.pth'

    net = net.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    print("loaded model")

    criterion = cal_loss

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            data, label = data['pointcloud'].to(device).float(), data['category'].to(device)

            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(data[0].cpu().detach().numpy())

            #o3d.visualization.draw_geometries([pcd])

            data = data.permute(0, 2, 1)
            print("data shape: ", data.shape)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            print("label: ", label)
            print("preds: ", preds)
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            print("loss: ", (test_loss / (batch_idx + 1)))
            print("accuracy: ", 100. * correct / total, correct, total)

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    print("loss", float("%.3f" % (test_loss / (batch_idx + 1))))
    print("acc", float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))))
    print("acc_avg", float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))))
    print("time", time_cost)
