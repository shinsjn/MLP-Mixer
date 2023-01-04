if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from mixer_model import *
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import torchvision
    from torchvision import transforms
    import wandb
    from tqdm import tqdm
    from utils import *

    # ----------- get param ------------

    hyperparam_defaults = dict(
        in_channels=3,
    hidden_size = 512,
    class_num = 1000,
    patch_size = 16,
    input_size = 224,
    layer_depth = 8,
    token_dim = 256,
    channel_dim = 2048,
    )

    run = wandb.init(config=hyperparam_defaults, project='MLP_Mixer')
    config = wandb.config
    # ----------- model prepare ------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Mixer(
        in_channels=config.in_channels,
        dim=config.hidden_size,
        class_num=config.class_num,
        patch_size=config.patch_size,
        input_size=config.input_size,
        layer_depth=config.layer_depth,
        token_dim=config.token_dim,
        channel_dim=config.channel_dim
    ).to(device)
    wandb.watch(model)

    print("==================== model info ======================")
    print(summary(model,(config.in_channels,config.input_size,config.input_size)))


    # ----------- data load ------------

    trans = transforms.Compose([transforms.Resize((config.input_size, config.input_size)),              # need ToTensor
                                transforms.ToTensor()])
    # trainset = torchvision.datasets.ImageFolder(
    #     'C:/Users/shins/PycharmProjects/MLP-MIXER/data/dataset/Fast Food Classification V2/Train', transform=trans)

    trainset = torchvision.datasets.ImageFolder(
        '/home/shinsjn/pycharm/MLP-MIXER/data/dataset/Fast Food Classification V2/Train', transform=trans)

    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True,pin_memory=True)

    # testset = torchvision.datasets.ImageFolder(
    #     'C:/Users/shins/PycharmProjects/MLP-MIXER/data/dataset/Fast Food Classification V2/Test', transform=trans)

    testset = torchvision.datasets.ImageFolder(
            '/home/shinsjn/pycharm/MLP-MIXER/data/dataset/Fast Food Classification V2/Test', transform=trans)

    testloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False,pin_memory=True)

    print("=============== data label ===============")
    print(trainset.class_to_idx)
    print(testset.class_to_idx)


    # ----------- train prepare ------------
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # -------------- train start --------------
    print("******************* train start *********************")

    for epoch in tqdm(range(config.epochs)):
        train_loss = 0
        test_loss = 0
        test_correct = 0

        for i,data in enumerate(trainloader):
            imgs,labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            if i % 100 == 0:
                test_loss = 0
                model.eval()
                with torch.no_grad():
                    for test_imgs, test_labels in testloader:
                        test_imgs = test_imgs.to(device)
                        test_labels = test_labels.to(device)
                        test_outputs = model(test_imgs)
                        test_loss+= criterion(test_outputs,test_labels).item()
                        test_correct += (test_outputs.argmax(1) == test_labels).type(torch.float).sum().item()

                metrics = {'test_accuracy': test_correct/len(testloader.dataset), 'train_loss': train_loss / len(trainloader), 'test_loss': test_loss / len(testloader)}
                wandb.log(metrics)










