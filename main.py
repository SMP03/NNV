# Program by @SMP03 (https://github.com/SMP03)
# Third-party libraries
import pickle as pck
import random
import tkinter as tk
from sys import exit
from tkinter import ttk

# Import start window and visual widgets
import Showcase
import StartMenu

# Create the starter window and assign net to myNet
starter_window = StartMenu.StartWindow()
myNet = starter_window.net
data_path = starter_window.database_path

# Main Window
root = tk.Tk()
root.geometry("{0}x{1}+0+0".format(1000, 665))  # Window Size
root.resizable(width=False, height=False)       # Avoid resizing by user
root.title("MNIST Neural Network")              # Window Title
root.protocol("WM_DELETE_WINDOW", exit)         # Set cross button to end program

# Load training data
with open(data_path, "rb") as file:
    data, data_labels, a, c = pck.load(file)

# Number of partitions of each epoch (Used for getting accuracy readings mid-epoch)
steps = 4

# Image Browser
img = Showcase.ImageBrowser(root, [5, 3], data, data_labels, myNet)
img.tk_widget.grid(row=0, column=0, columnspan=2)

# Accuracy Graphic
acc = Showcase.AccuracyGraph(root, [5, 3], steps, myNet.accuracy)
acc.tk_widget.grid(row=0, column=2, columnspan=2)

# Network Structure Visualizer
nns = Showcase.NetworkStructure(root, [5, 3], myNet, limit=10)
nns.tk_widget.grid(row=1, column=0, columnspan=2)

# Image Importer
image = Showcase.ImageExtractor(root, [5, 3], img.set_data)
image.widget.grid(row=1, column=2, columnspan=2)

# Test current accuracy for net and update widgets
accuracy = [myNet.evaluate(c)/len(c)]
acc.accuracy = myNet.accuracy
acc.update()


def train(iterations):  # TODO: Enable train parameters customization through app
    """Function for training the net through the text box."""

    # Globals needed to get the global variables (otherwise they would have to be passed through args)
    global myNet
    global a
    global c
    global acc
    global nns
    global accuracy
    global steps

    if iterations is not None:  # Needed for error handling (user input)
        for j in range(iterations):
            n_train = len(a)
            random.shuffle(a)  # Shuffle (reorganize randomly) training data

            # Separate training data into batches
            mini_batch_size = 10
            mini_batches = [a[k:(k + mini_batch_size)] for k in range(0, n_train, mini_batch_size)]

            # Train net with SGD and update the widgets
            for i in range(steps):
                myNet.step_sgd(mini_batches[i*(len(mini_batches)//steps):(i+1)*(len(mini_batches)//steps)], 0.1, c)
                acc.accuracy = myNet.accuracy
                acc.update()
                img.NN = myNet
                img.refresh()

            nns.NN = myNet
            nns.update()


def save(filename):
    """Function for saving the net state."""
    if ".pkl" not in filename:
        filename += ".pkl"
    myNet.save(filename)


def process_entry(entry):
    """Parse the entry text and return value."""
    input_str = entry.get()
    try:
        iterations = int(input_str)
        if iterations is not None:
            return iterations
    except:
        print("Error Parsing")
        return None


# Train widgets frame
train_frame = ttk.LabelFrame(master=root, text="Entrena la xarxa:")
train_frame.grid(row=2, column=0, columnspan=1, rowspan=2, sticky="WENS", padx=5, pady=5)

epochs_label = tk.Label(master=train_frame, text="Número d'èpoques per entrenar:")
epochs_label.grid(row=0, column=0, sticky="WENS", padx=5, pady=5)

epochs_entry = tk.Entry(master=train_frame, width=30)
epochs_entry.grid(row=0, column=1, sticky="WENS", padx=5, pady=5)

train_button = tk.Button(master=train_frame, text="Entrena!", command=lambda: train(process_entry(epochs_entry)))
train_button.grid(row=0, column=3, columnspan=1, sticky="WENS", padx=20, pady=5)

# Save net widgets
save_frame = ttk.LabelFrame(master=root, text="Desa la xarxa en forma d'arxiu:")
save_frame.grid(row=2, column=3, columnspan=1, rowspan=2, sticky="WENS", padx=5, pady=5)

save_label = tk.Label(master=save_frame, text="Nom de l'arxiu:")
save_label.grid(row=0, column=0, columnspan=1, sticky="WENS", padx=5, pady=5)

filename_entry = tk.Entry(master=save_frame, width=45)
filename_entry.grid(row=0, column=1, columnspan=1, sticky="WENS", padx=5, pady=5)

save_button = tk.Button(master=save_frame, text="Guarda!", command=lambda: save(filename_entry.get()))
save_button.grid(row=0, column=2, columnspan=1, sticky="WENS", padx=5, pady=5)

# Start loop
root.mainloop()
