import torch

def load_save_checkpoint(checkpoint_path):
    # __import__("pudb").set_trace()
    print('load the model weight')
    model_checkpoint = torch.load(checkpoint_path)
    del model_checkpoint['optimizer']
    torch.save(model_checkpoint, checkpoint_path)
    print('save the model weight')

def change_model_weight(checkpoint_path):
    __import__("pudb").set_trace()
    print('load the model weight')
    model_checkpoint = torch.load(checkpoint_path)
    # del model_checkpoint['optimizer']
    state_dict = model_checkpoint['state_dict']
    model_conv1_weight = state_dict['model.conv1.weight']
    model_conv1_weight = model_conv1_weight.repeat(1, 3, 1, 1)
    state_dict['model.conv1.weight'] = model_conv1_weight
    model_checkpoint['state_dict'] = state_dict
    # torch.save(model_checkpoint, checkpoint_path)
    # print('save the model weight')

if __name__ == '__main__':
    load_save_checkpoint('iter_40000_hrnet.pth')
    load_save_checkpoint('iter_40000_upnet_beit.pth')
    load_save_checkpoint('iter_40000_upnet_swin_base.pth')
    # change_model_weight('/data/projects/pre_trained/best.pth')