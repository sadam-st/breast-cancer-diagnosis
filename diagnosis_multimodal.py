# General Libraries
import os
import gc
import matplotlib
from tqdm import tqdm
from time import time
import datetime as dtime
import warnings
import pandas as pd
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
matplotlib.use('Agg') 
plt.rcParams.update({'font.size': 16})

import multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from albumentations import Compose, RandomScale, ShiftScaleRotate, RandomResizedCrop
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data Augmentation for Image Preprocessing
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, RandomScale, Resize,
                            RandomBrightnessContrast, CoarseDropout, HueSaturationValue,
                            Blur, GaussNoise, Rotate, RandomResizedCrop, 
                            ShiftScaleRotate, ToGray)
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet34, resnet50
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, precision_recall_curve
import torchvision.models as models
from vit_pytorch import ViT
from efficientnet_pytorch import EfficientNet

# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'TEC_Breast_Cancer', '_wandb_kernel': 'aot'}



# Load the CSV file
train = pd.read_csv("D:/meta_data_b.csv")
main_dir = r'D:\positive_cases\balanced_dataset'


# List to store all new rows with paths
new_rows = []

# Define a mapping of file names to view directories
file_to_view = {
    '00000001': 'L__CC',
    '00000002': 'L__MLO',
    '00000003': 'R__CC',
    '00000004': 'R__MLO'
}

# Iterate over each ID in the train DataFrame
for _, row in train.iterrows():
    patient_id = row['ID']
    laterality = row['Laterality_of_lesion']
    
    patient_folder = os.path.join(main_dir, patient_id)

    # Check if the patient's folder exists and is a directory
    if os.path.isdir(patient_folder):
        # Find the paths for each of the four views and add a new row for each
        for file_name, view_dir in file_to_view.items():
            new_row = row.copy()
            # The path format assumes each ID folder directly contains the view files
            file_path = os.path.join(patient_folder, f"{file_name}.tif")

            # Add the path to the new row if it exists, otherwise use None
            new_row['path'] = file_path if os.path.isfile(file_path) else None
            
            # Update the IsDX column based on Laterality_of_lesion
            if laterality == 0:
                if file_name in ['00000001', '00000002']:
                    new_row['IsDX'] = 1
                else:
                    new_row['IsDX'] = 0
            elif laterality == 1:
                if file_name in ['00000003', '00000004']:
                    new_row['IsDX'] = 1
                else:
                    new_row['IsDX'] = 0
            elif laterality == 2:
                new_row['IsDX'] = 1

            # Append the new row to the list
            new_rows.append(new_row)

# Convert the list of new rows into a DataFrame
train = pd.DataFrame(new_rows)

# Keep the columns same as the original dataframe plus the 'path' column
columns_order = ["ID", "Cancer_history", "Density", "Family_risk", "Age", "IsDX", "Laterality_of_lesion", "path"]



train = train[columns_order]

# Save the updated dataframe
train.to_csv("train_path.csv", index=False)





# Seed
torch.manual_seed(10)
#set_seed()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Read in Data
train = pd.read_csv(r"C:\Users\teevino\train_path.csv")


# ----- GLOBAL PARAMS -----
vertical_flip = 0.5
horizontal_flip = 0.5
csv_columns = ["Cancer_history", "Report", "Density", "Family_risk", "Age", "Event"]
no_columns = len(csv_columns)
output_size = 1
# -------------------------


# Dataset Class
class TECDataset(Dataset):
    def __init__(self, dataframe, vertical_flip, horizontal_flip, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip

        if is_train:
            self.transform = Compose([
                RandomScale(scale_limit=0.2, p=0.5),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                RandomResizedCrop(height=224, width=224),
                ToTensorV2()
            ], p=1)
        else:
            self.transform = Compose([ToTensorV2()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['path'][index]

        #print('Path of an image', image_path)

        # Read the TIFF image
        with Image.open(image_path) as img:
            image = np.array(img)  # Convert PIL image to numpy array

        # Normalize and convert image to np.float32 if it is in uint16 format
        if image.dtype == np.uint16:
            image = (image / 65535).astype(np.float32)

        # If image is grayscale (2D), convert it to 3-channel (3D)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)  # Stack the grayscale channel 3 times

        # Perform transformations
        transformed = self.transform(image=image)
        transf_image = transformed['image']

        # Ensure image has 3 channels before passing it through the model
        if transf_image.shape[0] == 1:
            transf_image = transf_image.repeat(3, axis=0)

        csv_columns = ["Cancer_history", "Density", "Family_risk", "Age", "IsDX"]
        
        
        categorical_values = self.dataframe.iloc[index][csv_columns[1:]].values
        categorical_values = [float(val) for val in categorical_values]  # Convert to float
        categorical_values_tensor = torch.tensor(categorical_values, dtype=torch.float32)

        target = torch.tensor(float(self.dataframe['IsDX'][index]), dtype=torch.float32)  # Convert target to tensor

        if self.is_train:
            return {
                "image": transf_image,
                "meta": categorical_values_tensor,
                "target": target
            }
        else:
            return {
                "image": transf_image,
                "meta": categorical_values_tensor
            }

def data_to_device(data):
    image = data['image'].to(DEVICE).to(torch.float32)
    metadata = data['meta'].to(DEVICE)  # Convert metadata to tensor and send to device
    targets = data['target'].to(DEVICE)
    return image, metadata, targets


# Sample data
sample_df = train.head(4)
# Instantiate Dataset object
dataset = TECDataset(sample_df, vertical_flip, horizontal_flip,
                      is_train=True)

# The Dataloader
#dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


# Output of the Dataloader
for k, data in enumerate(dataloader):
    image, meta, targets = data_to_device(data)
    image = image.to(torch.float32).to(DEVICE)
    meta = meta.to(DEVICE)
    print(f"Batch: {k}", "\n" +
      "Image:", image.shape, "\n" +
      "Meta:", meta, "\n" +
      "Targets:", targets, "\n" +
      "="*50)
    


#main method
#SENetowrks multimodal code

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        squeeze = x.view(batch_size, num_channels, -1).mean(dim=2)
        excitation = self.fc1(squeeze)
        excitation = nn.ReLU()(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        excitation = excitation.view(batch_size, num_channels, 1, 1)
        return x * excitation

class SEBlockResNet50(nn.Module):
    def __init__(self):
        super(SEBlockResNet50, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.se_layers = nn.ModuleList([SEBlock(256), SEBlock(512), SEBlock(1024), SEBlock(2048)])

    def forward(self, x):
        # Modify ResNet50 to include SE blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.se_layers[0](x)
        x = self.resnet.layer2(x)
        x = self.se_layers[1](x)
        x = self.resnet.layer3(x)
        x = self.se_layers[2](x)
        x = self.resnet.layer4(x)
        x = self.se_layers[3](x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNet50Network(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size

        # Define Feature part (IMAGE)
        self.features = SEBlockResNet50()  # Using modified ResNet50 with SE blocks
        
        # Metadata feature extraction part
        self.csv = nn.Sequential(
            nn.Linear(4, 400),  # Use 'no_columns' to match metadata dimension (4 in this case)
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # Define Classification part
        self.classification = nn.Linear(2048 + 400, output_size)  # Adjust the feature dimension if needed

    def forward(self, image, meta, prints=False):
    #def forward(self, image, prints=False):
        if prints: 
            print('Input Image shape:', image.shape)
            print('Input metadata shape:', meta.shape)
        
        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        meta = self.csv(meta)
        if prints: print('Meta Data:', meta.shape)
            
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # Apply final classification layer
        out = self.classification(image_meta_data)
        if prints: print('Out shape:', out.shape)

        return out




# Load Model
model_example = ResNet50Network(output_size=output_size, no_columns=no_columns).to(DEVICE)



# Outputs
out = model_example(image, meta, prints=True)

# Criterion example
criterion_example = nn.BCEWithLogitsLoss()
loss = criterion_example(out, targets.unsqueeze(1).float()) 
print("="*50)
print('Loss:', loss.item())        



def add_in_file(text, f):
    
    with open(f'logs_{VERSION}.txt', 'a+') as f:
        print(text, file=f)
        
        


#new code with testing
def train_folds(model, train_original):
    f = open(f"logs_{VERSION}.txt", "w+")
    
    
    #new code with train, valid and test
    train_original.to_csv('train_original.csv')

    # Calculate the number of rows for training and validation sets
    total_rows = len(train_original)
    train_size = int(total_rows * 0.6)  # 60% for training

# Split the DataFrame into training and remaining sets
    train_data = train_original.iloc[:train_size]  # First 70% of the DataFrame
    remain_data = train_original.iloc[train_size:]  # Remaining 30% of the DataFrame

# Calculate the number of rows for validation set (15% of the total data)
    valid_size = int(total_rows * 0.20)  # 20% for validation

# Split the remaining data into validation and test sets
    valid_data = remain_data.iloc[:valid_size]  # First 15% of the remaining 30%
    test_data = remain_data.iloc[valid_size:]  # Remaining 15% of the total data

# Reset index for all datasets
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

# Create Data instances for train, validation, and test
    train_dataset = TECDataset(train_data, vertical_flip, horizontal_flip)
    valid_dataset = TECDataset(valid_data, vertical_flip, horizontal_flip)
    test_dataset = TECDataset(test_data, vertical_flip, horizontal_flip)

# Dataloaders for train, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=WORKERS)

 
        
  
        
    # 🐝 W&B Tracking
    RUN_CONFIG = CONFIG.copy()
    params = dict(model=MODEL, 
              version=VERSION,
              epochs=EPOCHS, 
              batch=BATCH_SIZE1,
              lr=LR,
              weight_decay=WD)
    RUN_CONFIG.update(params)
   
    patience_f = PATIENCE

    # Optimizer/ Scheduler/ Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr = LR, 
                                     weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', 
                                      patience=LR_PATIENCE, verbose=True, factor=LR_FACTOR)
    criterion = nn.BCEWithLogitsLoss()


                                  
    #comulative_cm = np.zeros((2, 2), dtype=int)    
        # === EPOCHS ===
    for epoch in range(EPOCHS):
        start_time = time()
        correct = 0
        train_losses = 0

        model.train()
        train_probabilities = []
        true_labels = []

        # For each batch
        for k, data in tqdm(enumerate(train_loader)):
            # Move data to device (e.g., GPU or CPU)
            image, meta, targets = data_to_device(data)

            # Clear gradients first; very important
            optimizer.zero_grad()

            # Forward pass
            out = model(image, meta)
    
            # Compute loss
            loss = criterion(out, targets.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(out)

            # Collect probabilities and true labels
            train_probabilities.extend(probabilities.detach().cpu().numpy())
            true_labels.extend(targets.cpu().numpy())

        # Convert lists to numpy arrays
        train_probabilities = np.array(train_probabilities)
        true_labels = np.array(true_labels)

        # Convert probabilities to predicted class labels based on a threshold (e.g., 0.5)
        predicted_labels = (train_probabilities >= 0.5).astype(int)

        # Assuming train_data is a DataFrame and you need to save true labels
        train_data = pd.DataFrame({'IsDX': true_labels})
        train_data.to_csv('trainData.csv', index=False)

        # Compute different metrics for training
        train_acc = accuracy_score(train_data['IsDX'].values, predicted_labels)
        train_f1 = f1_score(train_data['IsDX'].values, predicted_labels, average='macro')
        train_recall = recall_score(train_data['IsDX'].values, predicted_labels, average='macro')
        train_roc = roc_auc_score(train_data['IsDX'].values, train_probabilities)  # average parameter not needed for binary classification
        train_prec = precision_score(train_data['IsDX'].values, predicted_labels, average='macro')

        
        
        
        model.eval()
        valid_preds = torch.zeros(size=(len(valid_data), 1), device=DEVICE, dtype=torch.float32)

        # Adjusted batch size handling
        with torch.no_grad():
            for k, data in tqdm(enumerate(valid_loader)):
                # Move data to device
                image, meta, targets = data_to_device(data)

                # Forward pass
                out = model(image, meta)
                #out = model(meta)
        
                # Apply sigmoid to get probabilities
                pred = torch.sigmoid(out)
        
                # Save predictions
                batch_size = image.shape[0]
                start_index = k * batch_size
                end_index = min(start_index + batch_size, len(valid_preds))
                valid_preds[start_index:end_index] = pred[:end_index - start_index]

        # Convert valid_preds to numpy array and flatten it
        valid_preds = valid_preds.cpu().numpy().flatten()

        # Ensure the lengths of valid_data and valid_preds are the same
        if len(valid_data) != len(valid_preds):
            min_length = min(len(valid_data), len(valid_preds))
            valid_data = valid_data[:min_length]
            valid_preds = valid_preds[:min_length]

        # Convert probabilities to predicted class labels based on a threshold (e.g., 0.5)
        predicted_labels = np.round(valid_preds)

        # Compute different metrics for validation
        valid_acc = accuracy_score(valid_data['IsDX'].values, predicted_labels)
        valid_f1 = f1_score(valid_data['IsDX'].values, predicted_labels, average='macro')
        valid_recall = recall_score(valid_data['IsDX'].values, predicted_labels, average='macro')
        valid_roc = roc_auc_score(valid_data['IsDX'].values, valid_preds)  # No average parameter needed for binary classification
        valid_prec = precision_score(valid_data['IsDX'].values, predicted_labels, average='macro')
        
         
   

        

        model.eval()
        test_preds = torch.zeros(size=(len(test_data), 1), device=DEVICE, dtype=torch.float32)

        # Adjusted batch size handling
        with torch.no_grad():
            for k, data in tqdm(enumerate(test_loader)):
                # Move data to device
                image, meta, targets = data_to_device(data)
                

                # Forward pass
                out = model(image, meta)
        
                # Apply sigmoid to get probabilities
                pred = torch.sigmoid(out)
        
                # Save predictions
                batch_size = image.shape[0]
                start_index = k * batch_size
                end_index = min(start_index + batch_size, len(test_preds))
                test_preds[start_index:end_index] = pred[:end_index - start_index]

        # Convert test_preds to numpy array and flatten it
        test_preds = test_preds.cpu().numpy().flatten()

        # Ensure the lengths of test_data and test_preds are the same
        if len(test_data) != len(test_preds):
            min_length = min(len(test_data), len(test_preds))
            test_data = test_data[:min_length]
            test_preds = test_preds[:min_length]

        # Convert probabilities to predicted class labels based on a threshold (e.g., 0.5)
        predicted_labels = np.round(test_preds)
        

        # Compute different metrics for testing
        test_acc = accuracy_score(test_data['IsDX'].values, predicted_labels)
        test_f1 = f1_score(test_data['IsDX'].values, predicted_labels, average='macro')
        test_recall = recall_score(test_data['IsDX'].values, predicted_labels, average='macro')
        test_roc = roc_auc_score(test_data['IsDX'].values, test_preds)  # No average parameter needed for binary classification
        test_prec = precision_score(test_data['IsDX'].values, predicted_labels, average='macro')
    
        predictions_df = pd.DataFrame(test_preds)
        

        test_data.to_csv('true_lables_test.csv')
        predictions_df.to_csv('predicted_labels_test.csv', index=False)
        
        # Compute ROC curve and ROC AUC
        fpr, tpr, _ = roc_curve(test_data['IsDX'].values, test_preds)
        roc_auc = roc_auc_score(test_data['IsDX'].values, test_preds)
        

    

        # Calculate time on Train + Eval
        duration = str(dtime.timedelta(seconds=time() - start_time))[:7]


        # PRINT INFO
        final_logs = '{} | Epoch: {:02d}/{} | Loss: {:.4f} | Acc_tr: {:.3f} | Acc_vd: {:.3f} | test_acc: {:.3f} | Train_ROC: {:.3f} | Valid_ROC: {:.3f} | Test_ROC: {:.3f} | Training_Prec: {:.3f} | Valid_Prec: {:.3f} | Test_Prec: {:.3f} | Training_F1: {:.3f} | Valid_F1: {:.3f} | Test_F1: {:.3f} | Train_Recall: {:.3f} | Valid_Recall: {:.3f} | Test_Recall: {:.3f} |'.\
            format(duration, epoch+1, EPOCHS, 
                    train_losses, train_acc, valid_acc, test_acc, train_roc, valid_roc, test_roc,
                    train_prec, valid_prec, test_prec, train_f1, valid_f1, test_f1, train_recall, valid_recall, test_recall)
        add_in_file(final_logs, f)
        print(final_logs)
        


        # === SAVE MODEL ===

        # Update scheduler (for learning_rate)
        scheduler.step(valid_roc)
        # Name the model
        model_name = f"Epoch{epoch+1}_ValidAcc{valid_acc:.3f}_ROC{valid_roc:.3f}.pth"
        
        torch.save(model.state_dict(), model_name)
        gc.collect()


EPOCHS = 30
PATIENCE = 10
WORKERS = 0
LR = 0.0005
WD = 0.0001
LR_PATIENCE = 2
LR_FACTOR = 0.5

BATCH_SIZE1 = 4           # for train
BATCH_SIZE2 = 4           # for valid

VERSION = 'v1'
MODEL = 'resnet50'
model1 = ResNet50Network(output_size=output_size, no_columns=no_columns).to(DEVICE)



# ------------------

# Constants and model instantiation...

if __name__ == '__main__':
    mp.freeze_support()

    # Call the train_folds function here
    train_folds(model=model1, train_original=train)


