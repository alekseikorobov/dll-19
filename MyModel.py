
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
import torch
import os, errno
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torchvision.models.detection as d

def get_instance_segmentation_model(num_classes) -> MaskRCNN:
    # # load an instance segmentation model pre-trained on COCO
    # #model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False,num_classes = num_classes)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

    # # get the number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # #in_features_mask = model.roi_heads.box_predictor.cls_score.in_channels
    
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # #model.roi_heads.box_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   #in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    # #                                                    hidden_layer,
    # #                                                    num_classes)
    # model.roi_heads.mask_predictor  = None
    
    trainable_backbone_layers = None
    rpn_score_thresh = None
    #веса использовать нельзя так как количество классов не подходит
    #weights=d.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    #weights_backbone = None
    data_augmentation  = 'hflip'
    kwargs = {"trainable_backbone_layers": trainable_backbone_layers}
    if data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if rpn_score_thresh is not None:
        kwargs["rpn_score_thresh"] = rpn_score_thresh
    
    kwargs['box_detections_per_img'] = 500
        
    model = maskrcnn_resnet50_fpn_v2(num_classes=num_classes, **kwargs)
    
    return model
    

def get_instance_segmentation_model_old(num_classes) -> MaskRCNN:
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False,num_classes = num_classes)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    #in_features_mask = model.roi_heads.box_predictor.cls_score.in_channels
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #model.roi_heads.box_predictor = MaskRCNNPredictor(in_features_mask,
                                                      #in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                    hidden_layer,
    #                                                    num_classes)
    model.roi_heads.mask_predictor  = None

    return model


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise
        
def save_model(model:MaskRCNN, epoch, lr, optimzer, save_dir, name = None):
    
    if name is None:
        name = f'FastRCNN_resnet50_{epoch}.pth'
    save_path = os.path.join(save_dir, name)
    
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
        
    print(f'Saving to {save_path}.')
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)
    return save_path

def remove_old_model(epoch, save_freq, save_dir):
    old_epoch = epoch - save_freq
    
    remove_path = os.path.join(save_dir, f'FastRCNN_resnet50_{old_epoch}.pth')
    
    if os.path.exists(remove_path):
        print(f'{remove_path=}.')
        os.remove(remove_path)
        
        
def load_model(model_path, num_classes, device)->MaskRCNN:
    model = get_instance_segmentation_model(num_classes)
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict['model'])
    return model