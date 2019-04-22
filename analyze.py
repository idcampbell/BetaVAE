import json
import multiprocessing
import time

import numpy as np
import PIL
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms


from model import BetaVAE
import utils


# Now import the parameters.
with open("params.json") as json_file:  
    params = json.load(json_file)
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
    cond = params["cond"]

    
# These are the filepaths that we will be using in the training and testing process.
test_csv = basepath + "CSVs/" + cond + "_test.csv"
heldout_csv = basepath + "CSVs/" + cond + "_heldout.csv"
data_dir = basepath + "Data/"
ckpt_dir = basepath + "Checkpoints/" + cond + "/"
log_dir = basepath + "Logs/" + cond + "/log.pkl"
output_dir = basepath + "Output/" + cond + "/"
    
    
# Generate n_random images using the model.
def generate(model, num, device):
    model.eval()
    z = torch.randn(num, z_dim).to(device)
    with torch.no_grad():
        return model.decode(z).cpu()

# returns the latent representation z for a given image tensor.
def get_z(im, model, device):
    model.eval()
    im = torch.unsqueeze(im, dim=0).to(device)
    with torch.no_grad():
        mu, var = model.encode(im)
        z = model.reparameterize(mu, var)
    return z, mu, var

def linear_interpolate(im1, im2, model, device):
    model.eval()
    z1, mu1, var1 = get_z(im1, model, device)
    z2, mu2, var2 = get_z(im2, model, device)
    factors = np.linspace(1, 0, num=10)
    result = []
    with torch.no_grad():
        for f in factors:
            z = (f * z1 + (1 - f) * z2).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)
    return result

def latent_arithmetic(im_z, attr_z, model, device):
    model.eval()
    factors = np.linspace(0, 1, num=10, dtype=float)
    result = []
    with torch.no_grad():
        for f in factors:
            z = im_z + (f * attr_z).type(torch.FloatTensor).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)
    return result

def plot_loss(train_loss, test_loss, filepath):
    train_x, train_l = zip(*train_loss)
    test_x, test_l = zip(*test_loss)
    plt.figure()
    plt.title('Train Loss vs. Test Loss')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.plot(train_x, train_l, 'b', label='train_loss')
    plt.plot(test_x, test_l, 'r', label='test_loss')
    plt.legend()
    plt.savefig(filepath)

if __name__ == "__main__":

    # Set the device and load in the model.
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = BetaVAE().to(device)
    
    # Load the last checkpoint for this model.
    model.load_last_model(ckpt_dir)
    
    # Set up the dataset and the DataLoader.
    heldout = pd.read_csv(heldout_csv)
    transform = transforms.Compose([transforms.ToTensor()])
    heldout_loader = DataLoader(heldout, batch_size=100, shuffle=True, num_workers=num_workers)
    
    # Now populate the DataFrame with the correct latent representations.
    test = pd.read_csv(test_csv)
    for index, row in df.iterrows():
        img = PIL.Image.open(test.loc[index].filename)
        tensor = transforms.functional.to_tensor(img)
        test.loc[index].latent_representation = get_z(tensor, model, "cpu")
    
    # Now iterate through the heldout images, calculate the loss, and save some reconstructions.
    heldout_loss = 0
    original_images = []
    rect_images = []
    with torch.no_grad():
        for batch_idx, data in enumerate(heldout_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            print("Loss for batch " + str(batch_idx) + ": " + loss) 
            heldout_loss += loss.item()
            if len(original_images) < 5:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())
    save_image(original_images + rect_images, output_dir + '.png', padding=0, nrow=len(original_images))
    average_loss = heldout_loss / len(heldout)
    print("Total heldout loss: " + heldout_loss)
    print("Average heldout loss: " + average_loss)
    
    # Generate images using the model and save them.
    samples = generate(model, 25, device)
    save_image(samples, output_dir + 'reconstructions.png', padding=0, nrow=5)

    train_losses, test_losses = utils.read_log(log_dir, ([], []))
    #plot_loss(train_losses, test_losses, PLOT_PATH)

    
    # Get images with the desired properties.
    

    # Interpolate between images latent arithmetic.
    #im1_z = get_z(im1, model, device)
    #im2_z = get_z(im2, model, device)
    #sunglass_z = get_average_z(man_sunglasses, model, device) - get_average_z(man, model, device)
    #arith1 = latent_arithmetic(man_z, sunglass_z, model, device)
    #arith2 = latent_arithmetic(woman_z, sunglass_z, model, device)

    #save_image(arith1 + arith2, OUTPUT_PATH + 'arithmetic-dfc' + '.png', padding=0, nrow=10)

    # Linear interpolate
    #inter1 = linear_interpolate(man[0], man[1], model, device)
    #inter2 = linear_interpolate(woman[0], woman_smiles[1], model, device)
    #inter3 = linear_interpolate(woman[1], woman_smiles[0], model, device)

    #save_image(inter1 + inter2 + inter3, OUTPUT_PATH + 'interpolate-dfc' + '.png', padding=0, nrow=10)