
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import esm # pre-trained ESM2 model

# Define encode function
def encode(molecule):
    # molecule is a numpy array of shape (N, 4), where N is the number of amino acids, and 4 corresponds to x, y, z, and type
    # use the ESM2 model to convert the molecule to an embedding vector of shape (1, 1280)
    esm2 = esm.pretrained.esm2_t6_43M_UR50S()
    embedding = esm2.to_embedding(molecule)
    return embedding

# Define decode function
def decode(embedding):
    # embedding is a tensor of shape (1, 1280), representing the embedding of a molecule
    # use the ESM2 model to convert the embedding to a molecule in the MorphProt format
    esm2 = esm.pretrained.esm2_t6_43M_UR50S()
    molecule = esm2.to_molecule(embedding)
    return molecule

# Load pre-trained model
model = torch.load('model.pt')
model.to(device)

# Define criterion, optimizer, and scheduler
criterion = nn.MSELoss() # use mean squared error loss to compare reaction effectiveness
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

        for target, reactant, product, effectiveness in data: # data is a list of tuples containing the target molecule, the reactant molecule, the product molecule, and the reaction effectiveness
            target = target.to(device)
            reactant = reactant.to(device)
            product = product.to(device)
            effectiveness = effectiveness.to(device)

            optimizer.zero_grad()

            # encode the target and the reactant molecules into vectors
            target_embedding = encode(target)
            reactant_embedding = encode(reactant)
            # concatenate the vectors into a vector of size 2560
            input_vector = torch.cat([target_embedding, reactant_embedding], dim=1)
            # pass the input vector to the model
            output = model(input_vector)
            # get the loss from the output and the true effectiveness
            loss = criterion(output, effectiveness)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # decode the output into a molecule
            output_molecule = decode(output)
            # compare the output molecule with the product molecule
            if np.array_equal(output_molecule, product):
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
