import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BATCH_SIZE = 100
EPOCHS = 10
LR = 0.0002
LATENT_DIM = 2  # Set to 2 for easy visualization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
# -------------------- CUDA INIT DONE ------------------------------
# Load MNIST and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize to [0, 1] range roughly
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#----------------------------- Dataset INIT DONE --------------------------------

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, LATENT_DIM)       # Mean
        self.fc_logvar = nn.Linear(400, LATENT_DIM)   # Log-Variance
        
        # Decoder
        self.fc3 = nn.Linear(LATENT_DIM, 400)
        self.fc4 = nn.Linear(400, 784)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

#---------------------------- VAE Architecture Done ----------------------------

def loss_function(recon_x, x, mu, logvar, use_kl=True):
    # 1. Reconstruction Loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if use_kl:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KLD = torch.tensor(0.0, device=DEVICE)

    return BCE + KLD, BCE, KLD

#------------------- Loss init done ------------------------------------------

def plot_latent_space(model, loader, title):
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            mu, logvar = model.encode(x.view(-1, 784))
            # We plot the mean vector (mu) as the representation
            all_z.append(mu.cpu().numpy())
            all_labels.append(y.numpy())
            
    all_z = np.concatenate(all_z, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.5, s=2)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True)
    plt.show()

def plot_generated_images(model, title):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(16, LATENT_DIM).to(DEVICE)
        sample = model.decode(z).cpu()
        
        # Reshape and plot
        plt.figure(figsize=(6, 6))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(sample[i].view(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

#---------------------------------------------------------------------- Visulazation

def train_model(use_kl, title_prefix):
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"--- Training {title_prefix} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(x)
            loss, bce, kld = loss_function(recon_batch, x, mu, logvar, use_kl=use_kl)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss / len(train_loader.dataset):.4f}")

    # Visualize Latent Space
    plot_latent_space(model, test_loader, f"Latent Space ({title_prefix})")
    
    # Generate Samples
    plot_generated_images(model, f"Generated Samples ({title_prefix})")
    
    return model


#---------------------------- Training DOne

# 1. Train WITHOUT KL Divergence (Standard Autoencoder behavior)
# This will show separate clusters but irregular gaps (bad for generation).
model_no_kl = train_model(use_kl=False, title_prefix="Without KL Divergence")

# 2. Train WITH KL Divergence (True VAE)
# This forces the latent space to be a smooth Gaussian (good for generation).
model_with_kl = train_model(use_kl=True, title_prefix="With KL Divergence")

print("\nComparison:\n1. Without KL: Latent space has gaps. Random sampling might pick 'empty' space, creating garbage images.\n2. With KL: Latent space is compact and continuous (Gaussian). Random sampling creates valid digits.")