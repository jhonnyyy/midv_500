#directory for saving the downloaded data
dataset_dir="./midv_500_dataset"

#learning rate 
lr=1e-4

#Batch Size 
batch_size=16

# Used in color Jittering
brightness=0.2
contrast=0.2 
saturation=0.2
hue=0.2

#Test size used in train test splitting 
test_size=0.2

num_epochs=20

output_document_path="./document_corner_model.pth"