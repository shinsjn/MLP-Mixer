if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from mixer_model import *

    # --------- base_model_param ---------
    in_channels = 3
    hidden_size = 512
    class_num = 1000
    patch_size = 16
    input_size = 224
    layer_depth = 8
    token_dim = 256
    channel_dim = 2048
    # ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Mixer(
        in_channels=in_channels,
        dim=hidden_size,
        class_num=class_num,
        patch_size=patch_size,
        input_size=input_size,
        layer_depth=layer_depth,
        token_dim=token_dim,
        channel_dim=channel_dim
    ).to(device)
    img = torch.rand(2, 3, 224, 224).to(device)
    output = model(img)
    print(output.shape)

    print(summary(model,(3,224,224)))
