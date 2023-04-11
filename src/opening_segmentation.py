import os
from cnn.corner.src.data_loader import get_loader
from cnn.corner.src.network import U_Net16
import torch
import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion, opening, closing, square
from cnn.semantic.src.data_set import open_dataset
from torch.utils.data import DataLoader

def segment_opening_corners(data_folder, images_path, im_fold, ret_corn_inference=False):
    ##CORNERS DETECTOR
    print("Predicting corners with corner detector CNN....")
    model = 'model_p4/v2'
    image_size = 256
    
    # model path
    model_path = '../weights/corners/models/' + model + '/' 
    model_path = model_path + os.listdir(model_path)[0]
    
    #To use the CNN model 
    test_loader = get_loader(image_path=images_path+im_fold,
                             image_size=image_size,
                             batch_size=1,
                             num_workers=0,
                             mode='test',
                             augmentation_prob=0.,
                             shuffle_flag=False)
    
    # load a trained model
    model = U_Net16()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.train(False)
    model.eval()
    
    #Dictionaries to storage prediction and images information    
    prediction_bin = {}
    prediction_rgb = {}
    bin_result_dict = {}
    progress = 0
    for ni, image in enumerate(test_loader):
        progress+=1
        print("Corner prediction process in {}%".format(progress*100/len(test_loader)))
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > .5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        image_name = test_loader.dataset.image_paths[ni].split('/')
        image_name = image_name[-1].split(".")
        image_name = image_name[0]
    
        corner = np.array(binary_result, dtype='uint8')
        prediction_bin[image_name] = corner*255
        
        cshp = corner.shape
        corner_rgb = np.zeros((cshp[0], cshp[1], 3), dtype='uint8')
        corner_rgb[:,:,2] = corner*255
        prediction_rgb[image_name] = corner_rgb

        bin_result_dict[image_name] = binary_result
    
    #Getting original images
    list_images = os.listdir(images_path+im_fold)
    images = {}
    for img in list_images:
        images[img[:-4]] = cv2.imread(images_path + im_fold + img)
    
    
    #Resizing predictions to the original size
    prediction_bin_resized = {}
    prediction_rgb_resized = {}
    for key in images:
        prediction_bin_resized[key] = cv2.resize(prediction_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv2.INTER_CUBIC)
        prediction_rgb_resized[key] = cv2.resize(prediction_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv2.INTER_CUBIC)
    
    #Overlaying prediction_rgb with original image
    overlayed_prediction = {}
    for key in images:
        overlayed_prediction[key] = cv2.addWeighted(images[key], 1.0, prediction_rgb_resized[key], 0.7, 0)
    
    #saving binary and overlayed predictions
    #Check if directory exists, if not, create it
    check_dir = os.path.isdir('../results/' + data_folder)
    if not check_dir:
        os.makedirs('../results/' + data_folder)
    progress=0
    for key in images:
        progress+=1
        print("Saving corner prediction process in {}%".format(progress*100/len(images)))
        #cv2.imwrite('../results/' + data_folder + '/' + key + '.png', images[key])
        cv2.imwrite('../results/' + data_folder + '/corners_' + key + '_overlayed.jpg', overlayed_prediction[key])
    
    if not ret_corn_inference:
    #Making prediction_bin_resized as binary
        progress=0
        for key in prediction_bin_resized:
            progress+=1
            print("Morphological operations in masks {}%".format(progress*100/len(prediction_bin_resized)))
            prediction_bin_resized[key] = prediction_bin_resized[key]>0
            prediction_bin_resized[key] = closing(prediction_bin_resized[key], disk(6)) 
        
        return images, prediction_bin_resized, overlayed_prediction
    
    else:
        corn_inference = {}
        for key in images:
            scale = np.array([images[key].shape[0],images[key].shape[1]])/image_size
            binary_result = bin_result_dict[key]            
            label_corners = label(binary_result)
            regions_corn = regionprops(label_corners)
            corn_inference[key] = np.array([scale * np.array(region.centroid) for region in regions_corn])
            ccc = np.array([scale * np.array(region.centroid) for region in regions_corn])


        return images, corn_inference, overlayed_prediction

def segment_opening(data_folder, images_path, im_fold, images=None, ret_op_inference=False):

    ##SEMANTIC SEGMENTATION OPENINNG DETECTOR    
    print("Predicting openings with semantic segmentation...")
       
    # model path
    model = 'model_p4/v1/'
    model_path = '../weights/semantic_openings/' + model
    model_path = model_path + os.listdir(model_path)[0]
    image_size = 256

    # model path
    test_ds=open_dataset(images_path + im_fold, transform='test')
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    # load a trained model
    model = U_Net16()
    device = torch.device('cpu')
    model=model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.eval()
    
    #Dictionaries to storage prediction and images information    
    prediction_bin = {}
    prediction_rgb = {}
    binary_result_dict = {} 
    for ni, image in enumerate(test_dl):
        image = image.to(device)
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > .5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        image_name = test_dl.dataset.path2imgs[ni].split('/')
        image_name = image_name[-1].split(".")
        image_name = image_name[0]
        
        corner = np.array(binary_result, dtype='uint8')
        prediction_bin[image_name] = corner*255
        
        cshp = corner.shape
        corner_rgb = np.zeros((cshp[0], cshp[1], 3), dtype='uint8')
        corner_rgb[:,:,2] = corner*255
        prediction_rgb[image_name] = corner_rgb

        binary_result_dict[image_name] = binary_result

    #Resizing predictions to the original size
    prediction_bin_resized = {}
    prediction_rgb_resized = {}
    
    if images is None:
        #Getting original images
        list_images = os.listdir(images_path+ im_fold)
        images = {}
        for img in list_images:
            images[img[:-4]] = cv2.imread(images_path + im_fold + img)

    for key in images:
        prediction_bin_resized[key] = cv2.resize(prediction_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv2.INTER_CUBIC)
        prediction_rgb_resized[key] = cv2.resize(prediction_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv2.INTER_CUBIC)
    
    #Overlaying prediction_rgb with original image
    overlayed_prediction = {}
    for key in images:
        overlayed_prediction[key] = cv2.addWeighted(images[key], 1.0, prediction_rgb_resized[key], 0.7, 0)
    
    progress=0
    for key in images:
        progress+=1
        print("Saving opening prediction process in {}%".format(progress*100/len(images)))
        cv2.imwrite('../results/' + data_folder + '/pred_op_bin_' + key + '.png', prediction_bin_resized[key])
        cv2.imwrite('../results/' + data_folder + '/' + key + '_op_overlayed.jpg', overlayed_prediction[key])
        
    #Making prediction_bin_resized as binary
    progress = 0
    for key in prediction_bin_resized:
        progress+=1
        print("Morphological operations in masks {}%".format(progress*100/len(prediction_bin_resized)))
        prediction_bin_resized[key] = prediction_bin_resized[key]>0

    if ret_op_inference:
        bboxes_dict = {}
        boxes_im_dict = {}
        labels_dict = {}
        label_op_f_dict = {}
        for key in images:
            scale = np.array([images[key].shape[0],images[key].shape[1]])/image_size
            binary_result = binary_result_dict[key]            
            label_op = label(binary_result)
            regions_op = regionprops(label_op)
            areas = np.array([region.area for region in regions_op])
            areas.sort()
            if len(areas)>1:
                area_min = .05*areas[-2]
            else:
                area_min = 0.
            regions_op_f = []
            bboxes = []
            labels = []
            boxes_im = np.copy(images[key])
            for reg in regions_op:
                if reg.area>=area_min:
                    regions_op_f.append(reg)        
            label_op_f = np.zeros(label_op.shape) #to save just regions of interest. Helps to skeleton_ransac. Find border
            for reg in regions_op_f:
                bx = reg.bbox
                bx_scaled = (int(bx[0]*scale[0]), int(bx[1]*scale[1]), int(bx[2]*scale[0]), int(bx[3]*scale[1]))
                bboxes.append(bx_scaled)
                labels.append('window') #In this architecture just are identified openings as windows
                cv2.rectangle(boxes_im, (bx_scaled[1],bx_scaled[0]), (bx_scaled[3], bx_scaled[2]), (0,0,255), 10)
                label_op_f += label_op==reg.label
            
            bboxes_dict[key] = bboxes
            boxes_im_dict[key] = boxes_im
            labels_dict[key] = labels
            label_op_f_dict[key] = cv2.resize(label_op_f, (int(image_size*scale[1]), int(image_size*scale[0])))

        return prediction_bin_resized, bboxes_dict, boxes_im_dict, labels_dict, label_op_f_dict
    else:
        return prediction_bin_resized