import torch
def get_model(model_name):
    if model_name=='UNet_adp':
        import model_definitions.unet_adaptive_activation as model
        model = model.Model(input_channels=1,output_channels=1,filter_bank=32)

    elif model_name=='UNeXt_s':
        import model_definitions.UNeXt as model
        model = model.Model_s(num_classes=1,input_channels=1, drop_rate=0.2)
    
    elif model_name=='UNeXt_m':
        import model_definitions.UNeXt as model
        model = model.Model(num_classes=1,input_channels=1, drop_rate=0.2)
    
    elif model_name=='UNet':
        import model_definitions.original_unet as model
        model = model.Model(in_channels=1,out_channels=1)

    elif model_name=='vgg16':
        import model_definitions.encoder_unet as model
        model = model.unet_vgg16(n_classes=1,pretrained=False, batch_size=32).type(torch.cuda.HalfTensor)
    
    elif model_name=='vgg19':
        import model_definitions.encoder_unet as model
        model = model.unet_vgg19(n_classes=1,pretrained=False, batch_size=32)     
    
    elif model_name=='Unetplusplus':
        import model_definitions.NestedUNet as model
        model = model.Model(num_classes=1,input_channels=1)
    
    elif model_name=='AttUnet':
        import model_definitions.attention_unet as model
        model = model.Model(img_ch=1,output_ch=1)
    
    elif model_name=='ResUnet':
        import model_definitions.ResUnet as model
        model = model.Model(channel=1)
    
    elif model_name=='ResUnet_plusplus':
        import model_definitions.ResUnet_plusplus as model
        model = model.Model(channel=1)



    return model