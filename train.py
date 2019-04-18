import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import models
import utils
from torchvision.utils import save_image

from dataset import ShapesDataset
from model import BetaVAE

with open('params.json') as json_file:  
    params = json.load(json_file)
    
    # Now import the parameters.
    dataset = params["dataset"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    test_batch_size = params["test_batch_size"]
    epochs = params["epochs"]
    seed = params["seed"]
    use_cuda = params["use_cuda"]
    display_step = params["display_step"]
    save_step = params["save_step"]
    basepath = params["base_path"]
    train_csv = basepath + params["train_csv"]
    test_csv = basepath + params["test_csv"]
    data_dir = basepath + params["data_dir"]
    ckpt_dir = basepath + params["ckpt_dir"]
    log_dir = basepath + params["log_dir"]
    output_dir = basepath + params["output_dir"]
    
# Function to manage the training of the network.
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(time.ctime(time.time()), epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss

# Function to manage the testing of the network.
def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0
    original_images = []
    rect_images = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()
            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))
    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)
    if return_images > 0:
        return test_loss, original_images, rect_images
    return test_loss

# Set some seeds.
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} if use_cuda else {}

# Set up GPU training and some training parameters.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Set up the dataset and the DataLoader.
transform = transforms.Compose([transforms.ToTensor()])
dataset = ShapesDataset(transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# CHANGE SHAPESDATASET so that we can load both training and test data.

# Instantiate the model and optimizer.
model = BetaVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Execute model training and testing.
if __name__ == "__main__":
    start_epoch = model.load_last_model(ckpt_dir) + 1
    train_losses, test_losses = utils.read_log(log_dir, ([], []))
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, device, loader, optimizer, epoch, display_step)
        test_loss, original_images, rect_images = test(model, device, loader, return_images=5)
        save_image(original_images + rect_images, output_dir + str(epoch) + '.png', padding=0, nrow=len(original_images))
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        utils.write_log(log_dir, (train_losses, test_losses))
        model.save_model(ckpt_dir + '%03d.pt' % epoch)