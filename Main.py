'''
Code to train a neural network to label organs
Implementation by Johannes Esch,

Unfortunately, I did not have enough time for visualisation in the estimated time.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nrrd
import os

# Define the neural network model
class OrganClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(OrganClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



# Function to load NRRD files
def load_nrrd(file_path):
    data, header = nrrd.read(file_path)
    return data


# Function to preprocess the data and labels
def preprocess_data(images_path, masks_path):
    images = []
    labels = []
    for image_file in os.listdir(images_path):
        if image_file.endswith('.nrrd'):
            image_path = os.path.join(images_path, image_file)
            mask_file = image_file.replace('image', 'mask')
            mask_path = os.path.join(masks_path, mask_file)
            if os.path.exists(mask_path):
                image_data = load_nrrd(image_path)
                mask_data = load_nrrd(mask_path)
                images.append(image_data.flatten())
                labels.append(get_organ_label(image_file))
    return np.array(images), np.array(labels)

# Function to get the organ label from the file name
def get_organ_label(file_name):
    organ_label = file_name.split('_')[1]
    organ_label_number = 0
    if organ_label == 'bonel1.nrrd' or organ_label == 'bonel2.nrrd':
        organ_label_number = 0
    elif organ_label == 'kidneyl.nrrd' or organ_label == 'kidneyr.nrrd':
        organ_label_number = 1
    elif organ_label == 'liver4.nrrd' or organ_label == 'liver7.nrrd' or organ_label == 'liver8.nrrd':
        organ_label_number = 2
    elif organ_label == 'musclel.nrrd' or organ_label == 'musclelr.nrrd':
        organ_label_number = 3
    elif organ_label == 'spleenl.nrrd' or organ_label == 'spleenr.nrrd':
        organ_label_number = 4

    return organ_label_number


# Define the paths to the image and mask directories
images_path = '/Users/johannesesch/Documents/Beruf/IKIM Challange/coding_challenge_mml/data/images'
masks_path = '/Users/johannesesch/Documents/Beruf/IKIM Challange/coding_challenge_mml/data/masks'

# Preprocess the data
images, labels = preprocess_data(images_path, masks_path)

# Define hyperparameters
input_size = images.shape[1]
hidden_size = 100
num_classes = 5
num_epochs = 100
batch_size = 16
learning_rate = 0.001

# Convert the data to PyTorch tensors
images = torch.from_numpy(images).float()
labels = torch.from_numpy(labels).long()



# Create a DataLoader for batch processing
dataset = torch.utils.data.TensorDataset(images, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = OrganClassifier(input_size, hidden_size, num_classes)

# Define the loss function and optimizer
cross = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = cross(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Save the Model
        if epoch % 5 == 0:
            torch.save(model, os.path.join('/Users/johannesesch/Documents/Beruf/IKIM Challange/coding_challenge_mml/Model_save/', 'epoch-{}.pt'.format(epoch)))


# Test the model

# Path to Test image and model
test_image_path = '/Users/johannesesch/Documents/Beruf/IKIM Challange/coding_challenge_mml/data/images/1_liver7.nrrd'
test_mask_path = '/Users/johannesesch/Documents/Beruf/IKIM Challange/coding_challenge_mml/data/masks/1_liver7.nrrd'

model_path = '/Users/johannesesch/Documents/Beruf/IKIM Challange/coding_challenge_mml/Model_save/epoch-95.pt'

def Test_model(test_image_path, test_mask_path, model_path ):
    #Load model
    test_model = torch.load(model_path)
    test_model.eval()

    # Load Test image
    test_image_data = load_nrrd(test_image_path)
    test_mask_data = load_nrrd(test_mask_path)

    test_input = torch.from_numpy(test_image_data.flatten()).float().unsqueeze(0)

    with torch.no_grad():
        test_model.eval()
        output = test_model(test_input)
        _, predicted_class = torch.max(output, 1)

    # prediction for test image
    predicted_label = {0: 'bone', 1: 'kidney', 2: 'liver', 3: 'muscle', 4: 'spleen'}
    predicted_organ = predicted_label[predicted_class.item()]

    print('Predicted organ: ', predicted_organ)

Test_model(test_image_path, test_mask_path, model_path)
