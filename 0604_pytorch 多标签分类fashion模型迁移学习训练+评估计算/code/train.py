import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # elesun
import argparse
from datetime import datetime
import torch
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from network import BuildModel
from validation import calculate_metrics, validate, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

def checkpoint_save(model, savedir, epoch):
    modelname = os.path.join(savedir, 'checkpoint-{:04d}.pth'.format(epoch))
    torch.save(model.state_dict(), modelname)
    print('Saved model:', modelname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--datapath', type=str, default="../../../../../datasets/cls/fashion", help="datasets path")
    parser.add_argument('--epochs', type=int, default=150, help="train epochs")
    parser.add_argument('--batch-size', type=int, default=160, help="train batch size") # elesun
    args = parser.parse_args()
    print("args\n",args)
    start_epoch = 1
    N_epochs = args.epochs
    batch_size = args.batch_size
    num_workers = 2  # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device: ", device)
    print("has", torch.cuda.device_count(), "GPUs")

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(os.path.join(args.datapath, 'styles.csv'))
    print("\nAll gender labels:\n", attributes.gender_labels)
    print("\nAll color labels:\n", attributes.color_labels)
    print("\nAll article labels:\n", attributes.article_labels)
    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = FashionDataset(os.path.join(args.datapath,'train.csv'), attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = FashionDataset(os.path.join(args.datapath,'val.csv'), attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BuildModel(n_color_classes=attributes.num_colors,
                             n_gender_classes=attributes.num_genders,
                             n_article_classes=attributes.num_articles)\
                            .to(device)
    optimizer = torch.optim.Adam(model.parameters())

    logdir = os.path.join('logs', datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    savedir = os.path.join('models', datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_color = 0
        accuracy_gender = 0
        accuracy_article = 0

        start = time.time()
        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_color, batch_accuracy_gender, batch_accuracy_article = \
                calculate_metrics(output, target_labels)

            accuracy_color += batch_accuracy_color
            accuracy_gender += batch_accuracy_gender
            accuracy_article += batch_accuracy_article

            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, color: {:.4f}, gender: {:.4f}, article: {:.4f}, time: {:.4f}s".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_color / n_train_samples,
            accuracy_gender / n_train_samples,
            accuracy_article / n_train_samples,
            time.time() - start))
        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)
        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)
        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)