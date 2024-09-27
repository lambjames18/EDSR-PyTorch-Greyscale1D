import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_stats(array):
    """Get the mean, 5th percentile, and 95th percentile of an array."""
    mean = np.nanmean(array)
    std = np.nanstd(array)
    p5 = np.nanpercentile(array, 5)
    p95 = np.nanpercentile(array, 95)
    return mean, p5, p95, std


def standardize_axis(ax, **kwargs):
    kwargs["labelsize"] = kwargs.get("labelsize", 20)
    kwargs["labelcolor"] = kwargs.get("labelcolor", "k")
    kwargs["direction"] = kwargs.get("direction", "in")
    kwargs["top"] = kwargs.get("top", True)
    kwargs["right"] = kwargs.get("right", True)
    ax.tick_params(axis="both", which="both", **kwargs)
    ax.grid(alpha=0.3, which="major")
    ax.grid(alpha=0.1, which="minor")


def make_legend(ax, **kwargs):
    # kwargs["bbox_to_anchor"] = kwargs.get("bbox_to_anchor", (1.03, 1.05))
    # kwargs["loc"] = kwargs.get("loc", "upper right")
    kwargs["fontsize"] = kwargs.get("fontsize", 18)
    kwargs["shadow"] = kwargs.get("shadow", True)
    kwargs["framealpha"] = kwargs.get("framealpha", 1)
    kwargs["fancybox"] = kwargs.get("fancybox", False)
    ax.legend(**kwargs)


def make_axis_log(ax, axis="x"):
    """Make an axis logarithmic."""
    if axis in ["x", "both"]:
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        xmin, xmax = ax.get_xlim()
        tick_range = np.arange(np.ceil(xmin), np.floor(xmax) + 1)
        tick_range_minor = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
        ax.xaxis.set_ticks(tick_range)
        minor_ticks = []
        for p in tick_range_minor:
            for x in np.linspace(10 ** p, 10 ** (p + 1), 10):
                if np.log10(x) >= xmin and np.log10(x) <= xmax:
                    minor_ticks.append(np.log10(x))
        ax.xaxis.set_ticks(minor_ticks, minor=True)
    if axis in ["y", "both"]:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ymin, ymax = ax.get_ylim()
        tick_range = np.arange(np.ceil(ymin), np.floor(ymax) + 1)
        tick_range_minor = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
        ax.yaxis.set_ticks(tick_range)
        minor_ticks = []
        for p in tick_range_minor:
            for y in np.linspace(10 ** p, 10 ** (p + 1), 10):
                if np.log10(y) >= ymin and np.log10(y) <= ymax:
                    minor_ticks.append(np.log10(y))
        ax.yaxis.set_ticks(minor_ticks, minor=True)


def mapColors(data, cmap, return_map=False, vrange=None):
    """Creates array of rgba values corresponding to a matplotlib colormap.

    Inputs
        data: array of values to be mapped to colors
        cmap: matplotlib cmap name"""
    cmap = plt.get_cmap(cmap)
    if vrange is not None:
        minima, maxima = vrange
    else:
        minima = data.min()
        maxima = data.max()
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    colors = cmap(norm(data))
    mappable = plt.cm.ScalarMappable(norm, cmap=cmap)
    if return_map == True:
        return colors, mappable
    else:
        return colors


class Viewer:
    def __init__(self, stack, cmap="viridis", vmin=None, vmax=None):
        if vmin is None:
            vmin = np.amin(stack)
        if vmax is None:
            vmax = np.amax(stack)
        self.slider_index = 0
        self.stack = stack
        self.dims = np.array(self.stack.shape)
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
        self.slice_slider = mpl.widgets.Slider(
            ax=axslice,
            label="Slice #",
            valmin=0,
            valmax=len(stack) - 1,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        # Create radio buttons for choosing slicing plane
        #axradio = plt.axes([right + 0.05, bot + 0.3, 0.05, height - 0.6])
        axradio = plt.axes([right + 0.05, bot, 0.05, height])
        axradio.set_title("Slicing plane")
        self.radio = mpl.widgets.RadioButtons(axradio, ("z", "y", "x"))
        self.radio.on_clicked(self.change_slicing_plane)

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
