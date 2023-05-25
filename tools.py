import os
import cv2
from deepface.commons import functions
import torchvision
import torch

def face_alignment(input_dir: str, output_dir: str):
    for file_name in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        detection = functions.extract_faces(img=img, enforce_detection=False)
        x, y, w, h = detection[0][1].values()
        aligned_img = img[int(y):int(y + h), int(x):int(x + w)]
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGRA2RGB)
        aligned_img = cv2.resize(aligned_img, (160, 160))
        cv2.imwrite(os.path.join(output_dir, file_name), aligned_img)


def embeddings_calc(gallery_paths: list, model, device, data_dir: str) -> dict:
    embeddings = {}
    for path in gallery_paths:
        image = cv2.imread(os.path.join(data_dir, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Resize((224, 224), antialias=True)(image)
        image = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        embeddings[path] = model(image).detach().cpu().numpy()
    return embeddings


# create a function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs, device, dataloaders, dataset_sizes):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                print('Training...')
            else:
                model.eval()  # Set model to evaluate mode
                print('Validating...')
            running_loss = 0.0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    return model

# create the data loaders
def create_dataloaders(data_dir: str, batch_size: int):
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224), antialias=True),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224), antialias=True),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
    }
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes
