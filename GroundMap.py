from messages import *
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class GroundMap():
    """
    Basic class for representing Ground Map for risk-aware planning
    Data entries:
    Header header # timestamp - gridmap timestamp
                  # frame_id - coordinate frame of the ground map 
    float64 resolution # the map resolution [m/cell]
    int width     # width of the map [cells]
    int height    # height of the map [cells]
    Pose pose     # the origin of the map [m, m, rad]. This is the real-world pose of the cell (0,0) in the map.
    Dict{str: np.array} layers # each layer represented as a NumPy array
    """

    def __init__(self, resolution, width, height, origin):
        self.layers = {}
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin = origin

    def populate_map(self, layer_sources):
        for layer_name, path in layer_sources.items():
            print("Creating ground map layer '" + layer_name + "' from file " + path)

            # Load the image file
            raw_data = mpimg.imread(path, "uint16")

            data = np.flip(raw_data, axis=0)
            data = (data * 65535)

            # # Handle different data types and normalize to [0, 1]
            # if raw_data.dtype == np.uint16:  # 16-bit data
            #     data = raw_data.astype(np.float64) # Normalize to [0, 1]
            # elif raw_data.dtype == np.float32:  # Float32 data
            #     data = raw_data  # Assuming it's already normalized to [0, 1]
            #     if data.max() > 1.0 or data.min() < 0.0:
            #         raise ValueError(f"Layer {layer_name}: Float32 data not in range [0, 1]")
            # else:
            #     raise ValueError(f"Layer {layer_name}: Unsupported data type {raw_data.dtype}")

            # Store the normalized data in the layer
            self.layers[layer_name] = data

            # # Plot the image
            # plt.figure(figsize=(8, 8))
            # plt.imshow(data, cmap='viridis', interpolation='nearest')
            # plt.colorbar(label='Pixel Intensity (Normalized)')
            # plt.title(f"Ground Map Layer: {layer_name}")
            # plt.axis('off')
            # plt.show()

            print("Created map layer [" + str(layer_name) + "] with min value " + str(np.min(data)) + " and max value " + str(np.max(data)))

    def ENUtoGrid(self, pos):
        return [int(i) for i in np.floor(np.array(pos) / self.resolution)]

    def GridToENU(self, pos):
        return (np.array(pos) + 0.5)  * self.resolution