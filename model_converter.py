import torch
import os

def update_state_dict(old_state_dict):
    new_state_dict = {}
    # Mapping from old layer names to new layer names
    conv_mapping = {
        'conv1': 'conv_layers.0',
        'conv2': 'conv_layers.1',
        'conv3': 'conv_layers.2',
        'conv4': 'conv_layers.3',
        'conv5': 'conv_layers.4',
        # Add more mappings if needed
    }
    fc_mapping = {
        'fc1': 'linear_layers.0',
        'fc2': 'linear_layers.1',
        'fc3': 'output_layer',
        # Add more mappings if needed
    }

    for old_key in old_state_dict.keys():
        new_key = old_key
        # Update the layer name in the key
        for old_layer_name, new_layer_name in {**conv_mapping, **fc_mapping}.items():
            if old_layer_name in old_key:
                new_key = old_key.replace(old_layer_name, new_layer_name)
                break
        # Add the updated key to the new state dictionary
        new_state_dict[new_key] = old_state_dict[old_key]

    return new_state_dict

def convert_checkpoint(old_checkpoint_path, new_checkpoint_path):
    # Load the old checkpoint
    old_state_dict = torch.load(old_checkpoint_path)

    # Update the state dict to match the new model's architecture
    updated_state_dict = update_state_dict(old_state_dict)

    # Save the updated state dict as a new checkpoint file
    torch.save(updated_state_dict, new_checkpoint_path)

# List of old model files
old_model_files = ['model_1965000.pt', 'model_1965000_target.pt']  # Replace with your file names
output_directory = 'converted_models'  # Directory to save converted models

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Convert each old model file
for old_model_file in old_model_files:
    new_model_file = os.path.join(output_directory, old_model_file)
    convert_checkpoint(old_model_file, new_model_file)
    print(f"Converted {old_model_file} to {new_model_file}")

#checkpoint = torch.load('model_1965000.pt')
#print(checkpoint)  # Or just print(checkpoint) to see the structure