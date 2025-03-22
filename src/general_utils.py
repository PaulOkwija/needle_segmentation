import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize


# Function for reading images
def read_image(img_path, img_size, color_scale):
    ''' Returns a numpy array of the image '''
    image = cv2.imread(img_path)
    image = Image.fromarray(image)
    image = image.resize((img_size[0], img_size[1]))
    image = np.array(image, dtype=np.uint8)

    if color_scale == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)
    elif color_scale == 'rgb':
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pass    

    return image


# Function for reading images
def generate_mask(orig_img_shape, req_img_shape, line_label):
    ''' Image should be an array '''
    # Convert it to a pillow image to draw a line on it
    mask = np.zeros(shape=orig_img_shape)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    draw.line(line_label, width=7, fill=255)
    
    mask = mask.resize(req_img_shape)
    mask = np.array(mask) == 255
    mask = mask.astype(int)
    mask = np.expand_dims(mask, axis=2)
    return mask

def plot_samples(df):
    # plt.subplots(3, 3, figsize=(8, 8))
    for k in range(5):
        image = df[k][0]
        image = image[0].unsqueeze(0)
        mask = df[k][1]
        plt.subplot(1, 2, 1)
        plt.axis(False)
        plt.imshow(image[0],cmap='gray')
        plt.subplot(1, 2, 2)
        plt.axis(False)
        plt.imshow(mask, cmap='gray')

        plt.tight_layout()
        plt.savefig("plots/sample_dataset_images_{}.png".format(k), dpi = 300)

def extract_df(path):
    print('Extracting data from all csv files in \'{}\' into a single dataframe...'.format(path))
    csv_files = sorted([x for x in list(os.listdir(path)) if x.endswith('.csv')])
    csv_files = list(map(lambda x: os.path.join(path, x), csv_files))

    line_labels = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df.rename(columns={'image_name':'Image_name','image':'Image_name','Image':'Image_name','Image_name':'Image_name'})
        line_labels = pd.concat([line_labels, df])


    line_labels = line_labels.drop_duplicates(subset='Image_name')
    line_labels = line_labels.reset_index(drop=True)
    return line_labels

def separate_df(df,types,state):
    for i in range(len(types)):
        print('Extracting {} dataframe'.format(types[i]))
        types[i] = df[df['Type']==types[i]]
        types[i] = types[i].sample(frac=1,random_state=state)
        types[i] = types[i].reset_index(drop=True)
    types[-1] = df[df.Type.isnull()].sample(frac=1,random_state=state).reset_index(drop=True)
    return tuple(types)

def get_images_masks_datasets(datasets, images_folder,ratios,verbose):
    ''' Takes in a dataframe containing the image names 
        and their line lables
    '''
    train_ratio = ratios[0]
    val_ratio = ratios[1]
    test_ratio = ratios[2]
    
    train_image_dataset = []
    train_mask_dataset = []
    test_image_dataset = []
    test_mask_dataset = []
    val_image_dataset = []
    val_mask_dataset = []

    tests_dict = {}

    for item in list(datasets.keys()):
        print("######### Loading {} dataset #########".format(item))
        line_labels = datasets[item]
        test_image = []
        test_mask = []
        image_size = (256,256)
        #take out only the first 300 0f every data frame
        data_length = len(line_labels)
        train_num = int(train_ratio*data_length)
        print('Train_num:',train_num)

        for i in range(train_num):
            image_data = line_labels.iloc[i]
            image_name = image_data.Image_name
            coordinates = image_data[['x1','y1','x2','y2']]

            # Read in the image
            image_path = os.path.join(images_folder, image_name)
            image = read_image(image_path, image_size,'gray')
            train_image_dataset.append(image)

            mask = generate_mask(image_size, image_size, coordinates)
            train_mask_dataset.append(mask)
        
        test_num = train_num + int(test_ratio*data_length)
        print('Test_num:',test_num-train_num)

        for i in range(train_num,test_num):
            image_data = line_labels.iloc[i]
            image_name = image_data.Image_name
            coordinates = image_data[['x1','y1','x2','y2']]

            # Read in the image
            image_path = os.path.join(images_folder, image_name)
            image = read_image(image_path, image_size,'gray')
            test_image_dataset.append(image)
            test_image.append(image)

            mask = generate_mask(image_size, image_size, coordinates)
            test_mask_dataset.append(mask)
            test_mask.append(mask)

        
        tests_dict[item] = [test_image,test_mask]

        val_num = test_num + int(val_ratio*data_length)
        print('Val_num:',data_length - test_num)
        # print("Data_length:",data_length)
        for i in range(test_num,data_length):
            image_data = line_labels.iloc[i]
            image_name = image_data.Image_name
            coordinates = image_data[['x1','y1','x2','y2']]

            # Read in the image
            image_path = os.path.join(images_folder, image_name)
            image = read_image(image_path, image_size,'gray')
            val_image_dataset.append(image)

            mask = generate_mask(image_size, image_size, coordinates)
            val_mask_dataset.append(mask)
        
        # image_dataset = np.array(image_dataset)
        # mask_dataset = np.array(mask_dataset)
    out = {"train images" : train_image_dataset, "Train masks" : train_mask_dataset, 
            "test images" : test_image_dataset, "test masks" : test_mask_dataset,
             "val images" : val_image_dataset, "val masks" : val_mask_dataset}
    
    

    keys = list(out.keys())

    for value in keys: 
        out[value] = np.float32(np.array(out[value]))

        if value.endswith("images"):
            out[value] = normalize(out[value], axis=1)
        
        if verbose:
            print("{} shape: ".format(value), out[value].shape)
            print("Minimum value: ",out[value][0].min())
            print("Maximum value: ", out[value][0].max())

    for value in list(tests_dict.keys()): 
        tests_dict[value][0] = np.float32(np.array(tests_dict[value][0]))
        tests_dict[value][1] = np.float32(np.array(tests_dict[value][1]))
        tests_dict[value][0] = normalize(tests_dict[value][0], axis=1)

    
    return (out[keys[0]], out[keys[1]], out[keys[2]], out[keys[3]], out[keys[4]], out[keys[5]],tests_dict)