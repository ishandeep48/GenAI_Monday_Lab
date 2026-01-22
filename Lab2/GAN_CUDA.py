import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# --- Configuration & Inputs ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_choice = input("Enter dataset (mnist/fashion): ").strip().lower()
epochs = int(input("Enter epochs (e.g., 50): "))
batch_size = int(input("Enter batch size (e.g., 64): "))
noise_dim = int(input("Enter noise dimension (e.g., 100): "))
lr = float(input("Enter learning rate (e.g., 0.0002): "))
save_interval = int(input("Enter save interval (e.g., 5): "))

os.makedirs('generated_samples', exist_ok=True)
os.makedirs('final_generated_images', exist_ok=True)

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Scale to [-1, 1]
])

if dataset_choice == 'fashion':
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
else:
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Models ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Initialize
netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# --- Training Loop ---
for epoch in range(1, epochs + 1):
    d_loss_total, g_loss_total, correct_d = 0, 0, 0
    
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_curr = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        real_labels = torch.ones(batch_curr, 1).to(device)
        fake_labels = torch.zeros(batch_curr, 1).to(device)

        # 1. Train Discriminator
        optimizerD.zero_grad()
        outputs = netD(real_imgs)
        loss_real = criterion(outputs, real_labels)
        
        noise = torch.randn(batch_curr, noise_dim).to(device)
        fake_imgs = netG(noise)
        outputs_fake = netD(fake_imgs.detach())
        loss_fake = criterion(outputs_fake, fake_labels)
        
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizerD.step()

        # 2. Train Generator
        optimizerG.zero_grad()
        outputs_g = netD(fake_imgs)
        loss_g = criterion(outputs_g, real_labels) # G wants D to think fakes are real
        loss_g.backward()
        optimizerG.step()

        d_loss_total += loss_d.item()
        g_loss_total += loss_g.item()
        correct_d += ((outputs > 0.5).sum().item() + (outputs_fake < 0.5).sum().item())

    # Logging
    d_acc = 100 * correct_d / (2 * len(dataset))
    print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss_total/len(dataloader):.4f} | D_acc: {d_acc:.2f}% | G_loss: {g_loss_total/len(dataloader):.4f}")

    # Save Samples
    if epoch % save_interval == 0:
        with torch.no_grad():
            sample_noise = torch.randn(25, noise_dim).to(device)
            samples = netG(sample_noise)
            save_image(samples, f"generated_samples/epoch_{epoch:02d}.png", nrow=5, normalize=True)

# --- Final Output & Classifier Prediction ---
print("\nGenerating final 100 images...")
with torch.no_grad():
    final_noise = torch.randn(100, noise_dim).to(device)
    final_imgs = netG(final_noise)
    for idx, img in enumerate(final_imgs):
        save_image(img, f"final_generated_images/img_{idx:03d}.png", normalize=True)

# Mock Classifier for Label Distribution (Simple CNN)
# In a real scenario, you'd load a weight file. Here we define a simple structure.
# Updated Mock/Pre-trained Classifier Logic

# Test with new code with maxpooling and better convo2d
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.main(x)

# Load weights if you have them, else it remains a 'stub' for the lab
classifier = SimpleClassifier().to(device)
if os.path.exists("mnist_classifier.pth"):
    classifier.load_state_dict(torch.load("mnist_classifier.pth"))
classifier.eval()

with torch.no_grad():
    # Convert [-1, 1] to [0, 1] for typical classifiers
    final_imgs_denorm = (final_imgs + 1) / 2 
    outputs = classifier(final_imgs_denorm)
    preds = outputs.argmax(dim=1)
    
    unique, counts = torch.unique(preds, return_counts=True)
    print("\n--- Predicted Label Distribution for 100 Generated Images ---")
    dist = dict(zip(unique.tolist(), counts.tolist()))
    for label in range(10):
        print(f"Label {label}: {dist.get(float(label), 0)} images")