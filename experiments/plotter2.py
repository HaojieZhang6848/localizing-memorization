
# filenames are of the form 'logs/mnist/resnet50_lr_0.01_noise_0.2_resnet50_cosine_seed_4_aug_0'
from gradients import get_all_grads
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

# datasets = ["mnist", "cifar10", "svhn"]
datasets = ["cifar10", "mnist", "svhn"]
# model_types = ["resnet9","resnet50", "vit"]
model_types = ["resnet9"]
# noise_types = [0.1, 0.2, 0.05]
# noise_types = [0.05]
noise_types = [0.1]
retraining_seeds = [4, 5, 6]


root = os.getcwd() + "/"

# for graphs on gradient similarity, we want all models for a given noise together. That is three subfigures for each dataset+noise rate
def plot_grads_helper(dir_name):
    # we need to get gradients for all the three directories
    #find names of other two directories
    seed_index = dir_name.rfind("seed_") + 5
    current_seed = int(dir_name[seed_index])
    dir_2 = dir_name[:seed_index] + str(current_seed + 1) + dir_name[seed_index + 1:]
    dir_3 = dir_name[:seed_index] + str(current_seed + 2) + dir_name[seed_index + 1:]

    gt, gc, gn = [], [], []
    for dir in [dir_name, dir_2, dir_3]:
        trackables_filename = os.path.join(root, dir, "trackables.pickle")
        if os.path.exists(trackables_filename):
            grads_total, grads_clean, grads_noisy = get_all_grads(trackables_filename)
            gt.append(grads_total)
            gc.append(grads_clean)
            gn.append(grads_noisy)


    #get the mean of the three directories
    grads_total = np.mean(gt, axis=0)
    grads_clean = np.mean(gc, axis=0)
    grads_noisy = np.mean(gn, axis=0)

    #get the std of the three directories
    std_total = np.std(gt, axis=0)
    std_clean = np.std(gc, axis=0)
    std_noisy = np.std(gn, axis=0)



    #return the average of the three directories across all epochs
    return np.mean(grads_total, axis=0), np.mean(grads_clean, axis=0), np.mean(grads_noisy, axis=0), np.mean(std_total, axis=0), np.mean(std_clean, axis=0), np.mean(std_noisy, axis=0)
    
def plot_combined_grads_figure(dataset, noise):
    #create a figure with three subfigures (3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    #get the three directories
    for model_type in model_types:
        dir_name = f'logs/{dataset}/{model_type}_lr_0.01_noise_{noise}_{model_type}_cosine_seed_4_aug_0'
        #get the average of the three directories across all epochs
        mean_total, mean_clean, mean_noisy, mean_std_total, mean_std_clean, mean_std_noisy = plot_grads_helper(dir_name)
        num_groups = len(mean_total)
        #plot the average of the three directories across all epochs
        axs[m].plot(range(num_groups), mean_total, color='blue')
        axs[m].plot(range(num_groups), mean_clean, color='green')
        axs[m].plot(range(num_groups), mean_noisy, color='red')

        #fill with std
        axs[m].fill_between(range(num_groups), mean_total - mean_std_total, mean_total + mean_std_total, color='blue', alpha=0.2)
        axs[m].fill_between(range(num_groups), mean_clean - mean_std_clean, mean_clean + mean_std_clean, color='green', alpha=0.2)
        axs[m].fill_between(range(num_groups), mean_noisy - mean_std_noisy, mean_noisy + mean_std_noisy, color='red', alpha=0.2)

        # add legend
        axs[m].legend(['Total', 'Clean', 'Noisy'])
        axs[m].set_xlabel('Model Layer Depth')
        axs[m].set_ylabel('Total Gradient Norm')
        axs[m].set_title(f'{model_type} {dataset} {noise}')
        axs[m].grid(b=True, which='major', color='#666666', linestyle='-')
        axs[m].minorticks_on()
        axs[m].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    plt.savefig(f'plots/grads_{dataset}_{noise}.pdf', bbox_inches='tight')


def plot_rewinding_helper(directory_name):
    #average over 3 seeds
    #find names of other two directories
    seed_index = directory_name.rfind("seed_") + 5

    current_seed = int(directory_name[seed_index])
    dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
    dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]

    if "resnet50" in directory_name:
        directory_name = dir_2

    #load the rewinding pickle
    c_a = []
    n_a = []

    dir_1 = directory_name
    for dir in [dir_1, dir_2, dir_3]:
        rwf = os.path.join(root, dir, "rewinding.pickle")
        if os.path.exists(rwf):
            with open(rwf,'rb') as f:
                rewinding_pickle  = pickle.load(f)
            trk_fname = os.path.join(root, dir, "trackables.pickle")
            # import ipdb; ipdb.set_trace()
            with open(trk_fname, 'rb') as f:
                trk = pickle.load(f)
                noise_mask = trk["noise_mask"]
   
            #final shape of outputs will be num_epochs x num_groups
            noisy_accs = np.zeros((len(rewinding_pickle), len(rewinding_pickle[0])))
            clean_accs = np.zeros((len(rewinding_pickle), len(rewinding_pickle[0])))

            for i in range(len(rewinding_pickle)):
                for j in range(len(rewinding_pickle[0])):
                    noisy_accs[i][j] = (np.array(rewinding_pickle[i][j]["acc_mask"]) == 1)[noise_mask==1].mean()
                    clean_accs[i][j] = (np.array(rewinding_pickle[i][j]["acc_mask"]) == 1)[noise_mask==0].mean()
            c_a.append(clean_accs)
            n_a.append(noisy_accs)
        else:
            print("no rewinding pickle found for ", dir)


    #get the mean of the three directories
    rewinding_clean = np.mean(c_a, axis=0)
    rewinding_noisy = np.mean(n_a, axis=0)
    #std
    rewinding_std_clean = np.std(c_a, axis=0)
    rewinding_std_noisy = np.std(n_a, axis=0)

    #rewinding is shape num_epochs x num_groups
    #we will plot a different line for each epoch [0,10,20,30...]
    #the colour of the lines should keep getting darker
    #the x axis should represent a group
    #the y axis should represent the accuracy
    
    return rewinding_clean, rewinding_noisy, rewinding_std_clean, rewinding_std_noisy



def plot_combined_rewinding_figure(dataset, noise):
    #row代表clean和noisy，column代表model type
    fig, axs = plt.subplots(2, len(model_types), figsize=(12, 6.5), squeeze=False)

    #get the three directories
    for m, model_type in enumerate(model_types):
        dir_name = f'logs/{dataset}/{model_type}_lr_0.01_noise_{noise}_{model_type}_cosine_seed_4_aug_0_cscore_0.0'
        #get the average of the three directories across all epochs
        rewinding_clean, rewinding_noisy, rewinding_std_clean, rewinding_std_noisy = plot_rewinding_helper(dir_name)
        num_lines = min(rewinding_clean.shape[0],6)
        num_groups = rewinding_clean.shape[1]
        #plot clean in row 0, and noisy in row 1
        for i, ex_type in enumerate(["clean", "noisy"]):
            rew = rewinding_clean if ex_type == "clean" else rewinding_noisy
            rew_std = rewinding_std_clean if ex_type == "clean" else rewinding_std_noisy
            #make a custom color bar of colors of green
            colorbar = np.zeros((num_lines, 3))
            for j in range(num_lines):
                colorbar[j] = (0, 1.0 - j/num_lines, 0) if ex_type == "clean" else (1.0 - j/num_lines, 0, 0)
            # axs[i][0].imshow(colorbar, aspect='auto', extent=[0, num_groups, 0, num_lines])

            #add circle markers for clean, and star markers for noisy 
            for j in range(0, num_lines):
                #get the mean of the next 10 epochs
                mean = rew[j]
                #also take the mean of std
                std = rew_std[j]
                #create a new figure using different shades of green colour. i do not want to use virdis colour palette.
                c =(0, 1.0 - j/num_lines, 0) if ex_type == "clean" else (1.0 - j/num_lines, 0, 0)
                axs[i][m].plot(range(num_groups), mean, color = c, label = j*10, marker = "o" if ex_type == "clean" else "*")
                #fill with std using different shades of green colour
                axs[i][m].fill_between(range(num_groups), mean - std, mean + std, color = c, alpha=0.2)

            #give heading "Epoch" to legend box and place it to the right of the figure outside it
            if m + 1 == len(model_types):
                axs[i][m].legend(title = "Epoch", loc = 'center left', bbox_to_anchor = (1, 0.5))
            if i == 1:
                axs[i][m].set_xlabel('Model Layer Depth')
            if m == 0:
                axs[i][m].set_ylabel('Accuracy')
            
            axs[i][m].set_title(f'{model_type.capitalize()} {ex_type.capitalize()}')
            axs[i][m].grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.2)
            # axs[i][0].minorticks_on()
            # axs[i][0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.savefig(f'plots/rewinding_{dataset}_{noise}.pdf', bbox_inches='tight')



def plot_combined_retraining_figure(dataset):
    # we will combine the retraining figures for all three noise levels into one figure (3 columns, 1 row)

    #get the three directories
    for m, noise in enumerate(noise_types):
        dir_name = f'{root}logs/{dataset}/resnet9_lr_0.01_noise_{noise}_resnet9_cosine_seed_4_aug_0_cscore_0.0'
        #get the average of the three directories across all epochs
        from plotter import plot_retraining_curves
        for seed in retraining_seeds:
            plot_retraining_curves(dir_name, f'plots/retraining_{dataset}_{noise}_seed_{seed}.pdf',seed=seed)

def plot_training_dynamic_helper(dir_name):
    #average over 3 seeds
    #find names of other two directories
    seed_index = dir_name.rfind("seed_") + 5
    current_seed = int(dir_name[seed_index])
    dir_2 = dir_name[:seed_index] + str(current_seed + 1) + dir_name[seed_index + 1:]
    dir_3 = dir_name[:seed_index] + str(current_seed + 2) + dir_name[seed_index + 1:]
    
    total_acc_all_epochs = []
    noisy_acc_all_epochs = []
    clean_acc_all_epochs = []

    for dir in [dir_name, dir_2, dir_3]:
        trk_fname = os.path.join(root, dir, "trackables.pickle")
        if os.path.exists(trk_fname):
            with open(trk_fname, 'rb') as f:
                trk = pickle.load(f)
            total_correct_all_epochs, clean_correct_all_epochs, noisy_correct_all_epochs = np.array(trk["total_correct_all_epochs"]), np.array(trk["clean_correct_all_epochs"]), np.array(trk["noisy_correct_all_epochs"])
            total_num_all_epochs, clean_num_all_epochs, noisy_num_all_epochs = np.array(trk["total_num_all_epochs"]), np.array(trk["clean_num_all_epochs"]), np.array(trk["noisy_num_all_epochs"])
            total_acc_all_epochs.append(total_correct_all_epochs/total_num_all_epochs)
            noisy_acc_all_epochs.append(noisy_correct_all_epochs/noisy_num_all_epochs)
            clean_acc_all_epochs.append(clean_correct_all_epochs/clean_num_all_epochs)
        else:
            print("no trackables pickle found for ", dir)

    total_acc_all_epochs = np.mean(total_acc_all_epochs, axis=0)
    noisy_acc_all_epochs = np.mean(noisy_acc_all_epochs, axis=0)
    clean_acc_all_epochs = np.mean(clean_acc_all_epochs, axis=0)
    
    return total_acc_all_epochs, noisy_acc_all_epochs, clean_acc_all_epochs

def plot_training_dynamic_figure(dataset, noise):
    #create a figure with three subfigures (column num is as many as model types)
    fig, axs = plt.subplots(1, len(model_types), figsize=(12, 4), squeeze=False)

    #get the three directories
    for m, model_type in enumerate(model_types):
        dir_name = f'logs/{dataset}/{model_type}_lr_0.01_noise_{noise}_{model_type}_cosine_seed_4_aug_0_cscore_0.0'
        #get the average of the three directories across all epochs
        total_acc_all_epochs, noisy_acc_all_epochs, clean_acc_all_epochs = plot_training_dynamic_helper(dir_name)
        num_epochs = len(total_acc_all_epochs)
        
        if m+1 == len(model_types):
            #plot the average of the three directories across all epochs
            axs[0][m].plot(range(num_epochs), total_acc_all_epochs, color='blue')
            axs[0][m].plot(range(num_epochs), clean_acc_all_epochs, color='green')
            axs[0][m].plot(range(num_epochs), noisy_acc_all_epochs, color='red')

            # add legend
            axs[0][m].legend(['Total', 'Clean', 'Noisy'])
            axs[0][m].set_xlabel('Epoch')
            axs[0][m].set_ylabel('Accuracy')
            axs[0][m].set_title(f'{model_type} {dataset} {noise}')
            axs[0][m].grid(visible=True, which='major', color='#666666', linestyle='-')
            axs[0][m].minorticks_on()
            axs[0][m].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    plt.savefig(f'plots/training_dynamic_{dataset}_{noise}.pdf', bbox_inches='tight')

for dataset in datasets:
    for noise in noise_types:
        try:
            plot_training_dynamic_figure(dataset, noise)
        except:
            pass
        try:
            plot_combined_rewinding_figure(dataset, noise)
        except:
            pass

for dataset in datasets:
    try:
        plot_combined_retraining_figure(dataset)
    except:
        pass