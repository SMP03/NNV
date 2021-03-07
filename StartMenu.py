import NN   # Neural Network Code

# Third-party libraries
import tkinter as tk
from tkinter import filedialog
from sys import exit
from tkinter import ttk


class StartWindow:
    def __init__(self):
        # Set window title, size, forbid resizing, assign exit program to cross button
        self.root = tk.Tk()
        self.root.title("Visualitzador de Xarxes Neuronals")
        self.root.resizable(width=False, height=False)
        self.root.protocol("WM_DELETE_WINDOW", exit)

        self.net = 0    # Placeholder

        self.database_path = None   # Placeholder

        self.create_widgets()   # Create widgets
        self.root.mainloop()    # Start loop

    def create_widgets(self):
        """Create all widgets. Position them through gridding."""
        # Title label
        self.title_label = tk.Label(self.root, text="Visualitzador de Xarxes Neuronals", font=("Arial Black", 14))
        self.title_label.grid(row=0, column=0, columnspan=3, sticky="WENS", padx=5, pady=5)

        # Select method parent frame
        self.select_frame = ttk.LabelFrame(self.root, text="Selecciona l'origen de la xarxa:")
        self.select_frame.grid(row=2, column=0, sticky="WENS", padx=5, pady=5)

        # Scratch generation parent frame
        self.scratch_frame = ttk.LabelFrame(self.select_frame, text="Des de zero:")
        self.scratch_frame.grid(row=1, column=0, columnspan=2, sticky="WENS", padx=5, pady=5)

        # Scratch prompt
        self.scratch_prompt = tk.Label(self.scratch_frame, text="Mida de les capes ocultes:")
        self.scratch_prompt.grid(row=0, column=0, columnspan=1, sticky="WENS", padx=5, pady=5)

        # Scratch layer sizes entry, bind <Return> to start generation
        self.scratch_entry = tk.Entry(self.scratch_frame)
        self.scratch_entry.grid(row=0, column=1, columnspan=1, sticky="WENS", padx=5, pady=5)
        self.root.bind('<Return>', lambda event: self.scratch_check())

        # Start generation
        self.scratch_button = tk.Button(self.scratch_frame, text="Genera!", command=self.scratch_check)
        self.scratch_button.grid(row=0, column=2, columnspan=1, sticky="WENS", padx=5, pady=5)

        # Select net from file parent frame
        self.file_frame = ttk.LabelFrame(self.select_frame, text="Des d'arxiu:")
        self.file_frame.grid(row=2, column=0, columnspan=2, sticky="WENS", padx=5, pady=5)

        # Button to open file explorer
        # TODO: had to bodge it to get the center text align
        self.file_button = tk.Button(self.file_frame,
                                     text="\t\t          Selecciona l'arxiu\t\t\t", command=self.file_check)
        self.file_button.grid(row=0, column=1, columnspan=1, sticky="WENS", padx=5, pady=5)

        # Select database from file parent frame
        self.data_frame = ttk.LabelFrame(self.root, text="Selecciona dades d'entrenament:")
        self.data_frame.grid(row=1, column=0, columnspan=2, sticky="WENS", padx=5, pady=5)

        # Button to open file explorer
        # TODO: had to bodge it to get the center text align
        self.data_button = tk.Button(self.data_frame,
                                     text="\t\t          Selecciona l'arxiu\t\t\t", command=self.data_check)
        self.data_button.grid(row=0, column=1, columnspan=1, sticky="WENS", padx=5, pady=5)

    def scratch_check(self):
        """Generate net with layer sizes."""
        input = self.scratch_entry.get()
        try:
            layers = [784]
            if input:
                for num in input.split(", "):
                    layers.append(int(num))
            layers.append(10)
            self.net = NN.NeuralNetwork(layers)
            print("Successful Generation")
            self.root.destroy()
        except:
            print("Failed")

    def file_check(self):
        """Open file explorer, get file path and load net"""
        filename = tk.filedialog.askopenfilename(title="Seleccioni l'arxiu",
                                                 filetypes=(("pickled files", "*.pkl"), ("all files", "*.*")))
        try:
            self.net = NN.load_net(filename)
            print("Successful Load")
            self.root.destroy()
        except:
            print("Failed")

    def data_check(self):
        """Open file explorer, get file path and return database location"""
        filename = tk.filedialog.askopenfilename(title="Seleccioni l'arxiu",
                                                 filetypes=(("pickled files", "*.pkl"), ("all files", "*.*")))
        try:
            self.database_path = filename
            self.data_button.configure(text=str(self.database_path).split("/")[-1])
        except:
            print("Failed")


