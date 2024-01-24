import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets

class Viewer:
    def __init__(self, stack, cmap, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.amin(stack)
        if vmax is None:
            vmax = np.amax(stack)
        self.slider_index = 0
        self.stack = stack
        #self.dims = np.array(self.stack.shape)
        plt.close(81234)
        self.fig = plt.figure(81234, figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Slicing plane: " + "zyx"[self.slider_index] + "-axis")
        # Show images
        self.im = self.ax.imshow(self.stack[0], cmap=cmap, vmin=vmin, vmax=vmax)
        # Put slider on
        plt.subplots_adjust(left=0.15, bottom=0.15)
        left = self.ax.get_position().x0
        bot = self.ax.get_position().y0
        height = self.ax.get_position().height
        right = self.ax.get_position().x1
        axslice = plt.axes([left - 0.15, bot, 0.05, height])
        self.slice_slider = matplotlib.widgets.Slider(
            ax=axslice,
            label="Slice #",
            valmin=0,
            valmax=len(stack) - 1,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        # Create radio buttons for choosing slicing plane
        '''axradio = plt.axes([right + 0.05, bot + 0.3, 0.05, height - 0.6])
        axradio.set_title("Slicing plane")
        self.radio = matplotlib.widgets.RadioButtons(axradio, ("z", "y", "x"))
        self.radio.on_clicked(self.change_slicing_plane)'''

        # Enable update functions
        self.slice_slider.on_changed(self.update_slice)
        plt.show()

    def update_slice(self, val):
        val = int(np.around(val, 0))
        image = self._create_slice(self.slider_index, val)
        self.im.set_data(image)
        self.im.axes.set_xlim(0, image.shape[1])
        self.im.axes.set_ylim(image.shape[0], 0)
        self.im.axes.figure.canvas.draw()
        self.fig.canvas.draw_idle()
    
    def change_slicing_plane(self, val):
        val = "zyx".index(val)
        self.slider_index = val
        self.slice_slider.valmax = self.dims[val] - 1
        self.slice_slider.ax.set_ylim(0, self.dims[val] - 1)
        self.ax.set_title("Slicing plane: " + "xyz"[self.slider_index] + "-axis")
        self.slice_slider.set_val(0)
        self.update_slice(0)
    
    def _create_slice(self, index, val):
        if index == 0:
            return self.stack[val]
        elif index == 1:
            return self.stack[:, val]
        elif index == 2:
            return self.stack[:, :, val]
