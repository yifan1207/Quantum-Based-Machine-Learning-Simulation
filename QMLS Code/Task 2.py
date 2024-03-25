import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import esm # pre-trained ESM2 model 
# Define mask functions
def mask(protein):
    # protein is a numpy array of shape (N, 4), where N is the number of amino acids, and 4 corresponds to x, y, z, and type
    # randomly select a bond or an amino acid to mask
    mask_type = np.random.choice(['bond', 'amino'])
    if mask_type == 'bond':
        # randomly select a bond index from 1 to N-1
        mask_idx = np.random.randint(1, protein.shape[0])
        # save the masked bond value
        mask_val = protein[mask_idx, :3]
        # replace the bond value with [MASK]
        protein[mask_idx, :3] = '[MASK]'
    else:
        # randomly select an amino acid index from 0 to N-1
        mask_idx = np.random.randint(0, protein.shape[0])
        # save the masked amino acid type
        mask_val = protein[mask_idx, 3]
        # replace the amino acid type with [MASK]
        protein[mask_idx, 3] = '[MASK]'
    return protein, mask_type, mask_idx, mask_val

# Define unmask function
def unmask(protein, mask_type, mask_idx, mask_val, output):
    # protein is a numpy array of shape (N, 4), where N is the number of amino acids, and 4 corresponds to x, y, z, and type
    # mask_type is either 'bond' or 'amino'
    # mask_idx is the index of the masked element
    # mask_val is the value of the masked element
    # output is a tensor of shape (1, 1280), representing the embedding of the masked element
    # use the ESM2 model to convert the output embedding to the corresponding bond value or amino acid type
    esm2 = esm.pretrained.esm2_t6_43M_UR50S()
    if mask_type == 'bond':
        # convert the output embedding to a bond value of shape (1, 3)
        bond_val = esm2.to_bond(output)
        # replace the [MASK] with the bond value
        protein[mask_idx, :3] = bond_val
    else:
        # convert the output embedding to an amino acid type of shape (1, 1)
        amino_type = esm2.to_amino(output)
        # replace the [MASK] with the amino acid type
        protein[mask_idx, 3] = amino_type
    return protein

# Load pre trained model
model = torch.load('model.pt')
model.to(device)

# Define criterion, optimizer, and scheduler
criterion = nn.MSELoss() # use mean squared error loss to compare embeddings
optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Define fine-tune function
def fine_tune(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for protein in data: # data is a list of numpy arrays representing protein structures
            protein = protein.to(device)

            optimizer.zero_grad()

            # mask a bond or an amino acid in the protein
            masked_protein, mask_type, mask_idx, mask_val = mask(protein)
            # pass the masked protein to the model
            output = model(masked_protein)
            # get the true embedding of the masked element from the ESM2 model
            true_embedding = esm2.to_embedding(mask_val)
            # get the loss from the output and the true embedding
            loss = criterion(output, true_embedding)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # unmask the protein and compare it with the original protein
            unmasked_protein = unmask(masked_protein, mask_type, mask_idx, mask_val, output)
            if np.array_equal(unmasked_protein, protein):
                running_corrects += 1

        scheduler.step()

        epoch_loss = running_loss / len(data)
        epoch_acc = running_corrects / len(data)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        print()

    print(f'Best Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Fine-tune the model
model = fine_tune(model, criterion, optimizer, scheduler, num_epochs=25)
