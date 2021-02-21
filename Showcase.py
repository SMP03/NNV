"""Imports==========================================================================================================="""
import tkinter as tk

import matplotlib
# Third party libs
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image
from pathlib import Path
from skimage import io, color, transform


class ImageBrowser(object):
    def __init__(self, root, inch_size, data, data_labels, NN, color_map="Greys", title=None, start_idx=0,
                 use_labels=True):
        # Figure, canvas and Tk handling
        matplotlib.use('TkAgg')
        self.figure = plt.figure(figsize=inch_size)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.figure, master=root)
        self.tk_widget = self.canvas.get_tk_widget()

        # Data assignment
        self.data = data
        self.data_labels = data_labels
        self.data_idx = start_idx

        # Image subplot
        self.img_subplot = self.figure.add_subplot(1, 2, 1)
        self.img_plot = plt.imshow(self.data[self.data_idx], cmap=color_map)
        plt.sca(self.img_subplot)
        plt.axis("off")

        # Text Label
        self.use_labels = use_labels
        self.label = plt.text(0.7, 0.1, str(self.data_labels[self.data_idx]), fontsize=20)

        # Back button
        self.backward_subplot = self.figure.add_subplot(15, 4, 57)
        self.bprev = matplotlib.widgets.Button(self.backward_subplot, 'Anterior', color="1.0", hovercolor="0.8")
        self.bprev.on_clicked(self.backward)

        # Forward button
        self.forward_subplot = self.figure.add_subplot(15, 4, 58)
        self.bnext = matplotlib.widgets.Button(self.forward_subplot, 'Següent', color="1.0", hovercolor="0.8")
        self.bnext.on_clicked(self.forward)

        # Bar graph subplot
        self.NN = NN
        self.bar_subplot = self.figure.add_subplot(1, 2, 2)
        self.nn_outputs = np.linspace(0, 1, 10)
        self.y_pos = np.arange(len(self.nn_outputs))
        self.update_graph(NN, self.data[self.data_idx].reshape((784, 1)))

        if title:
            self.img_subplot.set_title(title)

    def forward(self, *args):
        if self.data_idx + 1 >= len(self.data):
            self.data_idx = 0
        else:
            self.data_idx += 1
        self.refresh()

    def backward(self, *args):
        self.data_idx -= 1
        if self.data_idx == -1:
            self.data_idx = len(self.data) - 1
        self.refresh()

    def refresh(self, *args):
        self.img_plot.set_data(self.data[self.data_idx])
        if self.use_labels:
            self.label.set_text(str(self.data_labels[self.data_idx]))
        else:
            self.label.set_text("")
        self.update_graph(self.NN, self.data[self.data_idx].reshape((784, 1)))
        self.canvas.draw()

    def set_data(self, data, data_labels=False, data_index=0):
        self.data = data
        if data_labels:
            self.use_labels = True
            self.data_labels = data_labels
        else:
            self.use_labels = False

        self.data_idx = data_index
        self.refresh()

    def update_graph(self, NN, input):
        self.nn_outputs = NN.predict(input)
        self.bar_subplot.clear()
        bar_colors = ["grey" for i in range(10)]
        if self.use_labels:
            bar_colors[self.data_labels[self.data_idx]] = "limegreen"
        self.bar_subplot.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.bar_subplot.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.bar_subplot.set_xlim(0, 1)
        self.bar_subplot.grid(b=True, axis="x", which="both")
        self.bar_subplot.set_title("Outputs de la xarxa")
        self.bar_subplot.barh(self.y_pos, self.nn_outputs.transpose()[0], align='center', color=bar_colors)


class AccuracyGraph(object):
    def __init__(self, root, inch_size, steps, accuracy):
        matplotlib.use('TkAgg')
        self.figure = plt.figure(figsize=inch_size)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.figure, master=root)
        self.tk_widget = self.canvas.get_tk_widget()
        self.steps = steps

        self.subplot = self.figure.add_subplot(1, 1, 1)
        self.figure.subplots_adjust(bottom=0.15, left=0.20)

        self.accuracy = accuracy
        self.multiplier = 1


    def update(self):
        xval = np.arange(0, len(self.accuracy))
        self.subplot.clear()
        self.subplot.set_ylim(0, 1)
        self.subplot.plot(xval, self.accuracy, marker=".", linestyle="")

        x_locations = np.arange(0, len(self.accuracy), self.steps * self.multiplier)
        x_labels = np.arange(0, len(self.accuracy), self.steps * self.multiplier) // self.steps

        while len(x_locations) > 8:
            self.multiplier *= 2
            x_locations = np.arange(0, len(self.accuracy), self.steps * self.multiplier)
            x_labels = np.arange(0, len(self.accuracy), self.steps * self.multiplier) // self.steps

        plt.sca(self.subplot)
        plt.xticks(x_locations, x_labels)
        plt.grid()

        self.subplot.set(xlabel='Èpoques d\'entrenament',
                         ylabel=f'Precisió (Actual: {round(self.accuracy[-1] * 100)}%)')
        self.canvas.draw()


class NetworkStructure(object):
    def __init__(self, root, inch_size, NN, limit=10):
        matplotlib.use('TkAgg')
        self.figure = plt.figure(figsize=inch_size)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.figure, master=root)
        self.tk_widget = self.canvas.get_tk_widget()

        self.subplot = self.figure.add_subplot(1, 1, 1)
        self.figure.subplots_adjust(bottom=0.15, left=0.20)

        self.NN = NN
        self.limit = limit

        self.update()

    def update(self):
        self.subplot.clear()
        self.subplot.set_title("Estructura de la xarxa")
        plt.sca(self.subplot)
        for layer in range(len(self.NN.sizes)):
            self.draw_layer(layer, self.NN.sizes[layer])
        plt.axis('scaled')
        plt.axis('off')
        self.label = plt.text(0, -20, f"Mida de les capes: {self.NN.sizes}")
        self.canvas.draw()

    def draw_layer(self, n_layer, layer):
        for n_neuron in range(min(layer, self.limit)):
            self.draw_neuron(n_layer, n_neuron, np.mean(np.abs(self.NN.biases[n_layer-1])))

    def draw_neuron(self, n_layer, n_neuron, aveb):
        linw = 1  # Line width

        # Figure out neuron's x and y values in respect to the figure size
        figx, figy = self.figure.get_size_inches() * self.figure.dpi        # Figure size in pixels
        padx, pady = 10, 10                                 # Padding in pixels
        spcx = (figx - (2 * padx)) / (len(self.NN.sizes) - 1)        # Spacing between layers
        spcy = (figy - (2 * pady)) / (min(self.NN.sizes[n_layer] - 1, self.limit-1))    # Spacing between neurons
        neux = padx + (spcx * n_layer)                      # Neuron x pos
        neuy = pady + (spcy * n_neuron)                     # Neuron y pos

        neur = spcy / 4  # Neuron radius

        # If is first layer don't try to draw weights and bias
        if n_layer == 0:
            circ = "k"      # Circle color = black
            cira = 1.0      # Circle alpha = 1.0 (opaque)

        # If bias is positive set neuron color to blue; if negative set neuron color to red
        else:
            neub = self.NN.biases[n_layer-1][n_neuron]         # Neuron bias
            if neub > 0:
                circ = "b"  # Circle color = blue
            else:
                circ = "r"  # Circle color = red
            cira = float(1 - 4 * self.NN.sigmoid_prime((2 / aveb) * neub))  # Circle alpha as a relation

            neuw = self.NN.weights[n_layer-1][n_neuron]        # Neuron weights
            avew = np.mean(np.abs(neuw))  # Weight average
            for n_weight in range(min(len(neuw), self.limit)):
                prex = padx + (spcx * (n_layer-1))  # Previous neuron x pos
                prey = pady + ((figy - (2 * pady)) / (min(self.NN.sizes[n_layer-1]-1, self.limit-1)) * n_weight)  # Previous neuron y pos
                if neuw[n_weight] > 0:
                    linc = "b"  # Circle color = blue
                else:
                    linc = "r"  # Circle color = red
                lina = 1 - 4 * self.NN.sigmoid_prime((2 / avew) * neuw[n_weight])  # Line alpha as a relation

                line = plt.Line2D((prex, neux), (prey, neuy), linewidth=linw, color=linc, alpha=lina)
                line.set(zorder=10)
                self.subplot.add_line(line)

        circle = plt.Circle((neux, neuy), radius=neur, color=circ, alpha=cira)  # Plot circle
        circle.set(zorder=20)                                                   # Bring to top
        self.subplot.add_patch(circle)                                             # Draw circle


class WeightBrowser(object):    # Not used in main program
    def __init__(self, root, inch_size, NN, color_map="Greys", title=None, start_idx=0):
        # Figure, canvas and Tk handling
        matplotlib.use('TkAgg')
        self.figure = plt.figure(figsize=inch_size)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.figure, master=root)
        self.tk_widget = self.canvas.get_tk_widget()

        # Data assignment
        self.data_idx = start_idx
        self.NN = NN

        # Image subplot
        self.img_subplot = self.figure.add_subplot(1, 1, 1)
        self.img_plot = plt.imshow(self.NN.weights[0][self.data_idx].reshape(28, 28), cmap=color_map)
        self.img_subplot.set_title("Weight Visualizer")
        plt.sca(self.img_subplot)
        plt.axis("off")

        # Text Label
        self.label = plt.text(-7, 5, self.data_idx + 1, fontsize=20)

        # Back button
        self.backward_subplot = self.figure.add_subplot(15, 2, 29)
        self.bprev = matplotlib.widgets.Button(self.backward_subplot, 'Previous', color="1.0", hovercolor="0.8")
        self.bprev.on_clicked(self.backward)

        # Forward button
        self.forward_subplot = self.figure.add_subplot(15, 2, 30)
        self.bnext = matplotlib.widgets.Button(self.forward_subplot, 'Next', color="1.0", hovercolor="0.8")
        self.bnext.on_clicked(self.forward)

        if title:
            self.img_subplot.set_title(title)

    def forward(self, *args):
        self.data_idx += 1
        if self.data_idx == len(self.NN.weights[0]):
            self.data_idx = 0
        self.refresh()

    def backward(self, *args):
        self.data_idx -= 1
        if self.data_idx == -1:
            self.data_idx = len(self.NN.weights[0]) - 1
        self.refresh()

    def refresh(self, *args):
        self.img_plot.set_data(self.NN.weights[0][self.data_idx].reshape(28, 28))
        self.label.set_text(str(self.data_idx + 1))
        self.canvas.draw()


class ImageExtractor(object):
    def __init__(self, root, inch_size, data_send):
        self.root = root
        self.widget = tk.Frame(root)
        self.inch_size = inch_size

        self.image = None
        self.data_send = data_send

        self.panel = tk.Label(self.widget)
        self.panel.grid(row=1, column=2, rowspan=2, sticky="NWES")

        self.button = tk.Button(self.widget, text="Carregar imatge\n des d'arxiu", command=self.select_image)
        self.button.grid(row=1, column=1, rowspan=2, sticky="WE")

        self.update()

    def select_image(self):
        filename = tk.filedialog.askopenfilename(title="Select image")
        filename = Path(filename)

        self.set_image(filename)

    def set_image(self, filename):
        # Open image
        img = Image.open(filename)

        # Calculate size based on inch_size
        aspect_ratio = img.width / img.height
        print(aspect_ratio)
        dpi = self.root.winfo_fpixels('1i')
        pixels_y = round(dpi * self.inch_size[1])
        pixels_x = round(pixels_y * aspect_ratio)

        # Resize and create ImageTk object
        img = img.resize((pixels_x, pixels_y), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(img)

        # Set image
        self.panel.configure(image=self.image)
        self.cells = self.extract_numbers(filename)
        self.data_send(self.cells)

    def update(self):
        self.panel.configure(image=self.image)
        pass

    @staticmethod
    def extract_numbers(filename: "str") -> "list":
        # Load image from file
        image = io.imread(filename)

        # Convert to gray scale
        image = color.rgb2gray(image)

        # Divide the cells
        cells = []
        width = image.shape[1]
        height = image.shape[0]
        cell_x = width / 3
        cell_y = height / 5
        for y in range(5):
            for x in range(3):
                cell = image[round(y * cell_y):round((y + 1) * cell_y), round(x * cell_x):round((x + 1) * cell_x)]
                cells.append(cell)
                print(x, y)

        # Apply padding to exclude borders
        padding = 15
        for i in range(len(cells)):
            x, y = cells[i].shape[0], cells[i].shape[1]
            cell = cells[i][y // padding:(padding-1) * (y // padding), x // padding:(padding-1) * (x // padding)]
            # Colors to boolean (255=white, 0=black)
            cell = transform.resize(cell, (28, 28))
            cutoff = np.average(cell)
            mask_white = cell > cutoff
            mask_black = cell <= cutoff
            cell[mask_white] = 0.0
            cell[mask_black] = 1.0
            cells[i] = transform.resize(cell, (28, 28))
            print(cells[i].shape[0], cells[i].shape[1])

        return cells

