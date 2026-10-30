# Deep Visualization Toolbox

This is the code required to run the Deep Visualization Toolbox, as well as to generate the neuron-by-neuron visualizations using regularized optimization.
The toolbox and methods are described casually [here](http://yosinski.com/deepvis) and more formally in this paper:

 * Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. [Understanding neural networks through deep visualization](http://arxiv.org/abs/1506.06579). Presented at the Deep Learning Workshop, International Conference on Machine Learning (ICML), 2015.

If you find this paper or code useful, we encourage you to cite the paper. BibTeX:

    @inproceedings{yosinski-2015-ICML-DL-understanding-neural-networks,
    Author = {Jason Yosinski and Jeff Clune and Anh Nguyen and Thomas Fuchs and Hod Lipson},
    Booktitle = {Deep Learning Workshop, International Conference on Machine Learning (ICML)},
    Title = {Understanding Neural Networks Through Deep Visualization},
    Year = {2015}}



# Features

The main toolbox window looks like this, here showing a convolutional unit that responds to automobile wheels:

![DeepVis Toolbox Screenshot bus](doc/example_caffenet-yos_bus_wheel_unit.jpg?raw=true)

For a quick tour of the toolbox features, including what each pane of the above interface is showing, watch this [4 min YouTube video](https://www.youtube.com/watch?v=AgkfIQ4IGaM). In addition to processing images files from disk, the toolbox can run off a webcam for live network visualization **(below left)**.
The toolbox comes bundled with the default [caffenet-yos](models/caffenet-yos) model weights and pre-computed per-unit visualizations shown in the paper. Weights, but not per-unit visualizations, for [bvlc-googlenet](models/bvlc-googlenet) **(below right)** and [squeezenet](models/squeezenet) can be downloaded by scripts in their respective directories.

[![DeepVis Toolbox Screenshot webcam](doc/example_caffenet-yos_webcam.300.jpg)](doc/example_caffenet-yos_webcam.jpg?raw=true)
[![DeepVis Toolbox Screenshot bvlc-googlenet](doc/example_bvlc-googlenet_bus.300.jpg)](doc/example_bvlc-googlenet_bus.jpg?raw=true)

You can visualize your own model as well. However, note that the toolbox provides two rather separate sets of features; the first is easy to use with your own model, and the second is more involved:

1. **Forward/backward prop**: Images can be run forward through the network to visualize activations, and derivatives of any unit with respect to any other unit can be computed using backprop. In addition to traditional backprop, deconv from [Zeiler and Fergus (2014)](https://scholar.google.com/scholar?q=Zeiler+Visualizing+and+understanding+convolutional+networks) is supported as a way of flowing information backwards through the network. Doing forward and backward passes works for any Caffe model (including yours!) whose layer types `torchcaffe` supports.

2. **Per-unit visualizations**: Three types of per-unit visualizations can be computed for a network &mdash; max image, deconv of max image, activation maximization via regularized optimization &mdash; but these visualizations must be computed *outside* the toolbox and saved as jpg. The toolbox then loads these jpgs to display alongside units as they are selected. Visualizations must be pre-computed because they are far too expensive to run live. For example, going through the 1.3m image training set to find the images causing top-9 activations took 40 hours on our system (for all units). Per-unit visualization jpgs are provided for the caffenet-yos model, but not for the bvlc-googlenet or squeezenet models (and not for yours, but you can [compute them yourself](doc/computing_per_unit_visualizations.md)).

Summary:

| Model          | Forward/Backward prop | Per-unit visualizations |
| -------------  | ---------------- | ----------------------- |
| [caffenet-yos](models/caffenet-yos) | **easy**             | **included**            |
| [bvlc-googlenet](bvlc-googlenet)    | **easy**             | not-included, [generate](doc/computing_per_unit_visualizations.md) if desired            |
| [squeezenet](models/squeezenet)     | **easy**             | not-included, [generate](doc/computing_per_unit_visualizations.md) if desired            |
| your network | **easy** (just point to your model in `settings_local.py`)  | not-included, [generate](doc/computing_per_unit_visualizations.md) if desired  |



# Setting up and running the toolbox

This version of the toolbox runs on Python 3 and [PyTorch](https://pytorch.org/) instead of a compiled Caffe. The bundled `torchcaffe` package reads Caffe `.prototxt` network definitions and `.caffemodel` weights directly and provides the parts of the pycaffe API the toolbox needs, including backprop and deconv ([Zeiler and Fergus, 2014](https://scholar.google.com/scholar?q=Zeiler+Visualizing+and+understanding+convolutional+networks)). There is nothing to compile, and it runs on macOS (including the Apple GPU via MPS), Linux and Windows. The original Caffe models and their pre-computed visualizations work unchanged.

### Step 1: Install uv

All dependencies live in a project-local virtual environment managed by [uv](https://docs.astral.sh/uv/); nothing is installed system-wide. If you don't have uv yet, see the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`).

### Step 2: Get the code and install dependencies

    $ git clone https://github.com/cannonkalra/deep-visualization-toolbox
    $ cd deep-visualization-toolbox
    $ uv sync

`uv sync` downloads Python 3.12 if needed and creates `.venv/` with numpy, scipy, scikit-image, OpenCV, PyTorch and protobuf.

### Step 3: Configure and download a model

Create a `settings_local.py` from the template for the default `caffenet-yos` model:

    $ cp models/caffenet-yos/settings_local.template-caffenet-yos.py settings_local.py

*Note on settings:* Settings are split into two files: a versioned `settings.py` file that provides documentation and default values for all settings and an unversioned `settings_local.py` file. This latter file allows you to override any default setting to tailor the toolbox to your specific setup (CPU vs. GPU, webcam device, etc) and model (model weights, prototxt, sizes of the various panels shown in the toolbox, etc). This also makes it easy to distribute settings tweaks alongside models: for example, `models/bvlc-googlenet/settings_local.template-bvlc-googlenet.py` includes the appropriate window pane sizes and so on for the `bvlc-googlenet` model. To load a new model, just change the details in `settings_local.py`, perhaps by copying from the included template.

Then download the default model weights and corresponding top-9 visualizations saved as jpg (downloads a 230MB model and 1.1GB of jpgs to show as visualization):

    $ cd models/caffenet-yos/
    $ ./fetch.sh
    $ cd ../..

### Step 4: Run it!

    $ uv run ./run_toolbox.py

Once the toolbox is running, push 'h' to show a help screen. You can also have a look at `bindings.py` to see what the various keys do. If the window is too large or too small for your screen, set the `global_scale` and `global_font_size` variables in `settings_local.py` to values smaller or larger than 1.0.

By default the network runs on the GPU when one is available (Apple MPS or CUDA); set `caffevis_mode_gpu = False` in `settings_local.py` to use the CPU. The webcam is used when available (on macOS, allow camera access for your terminal when prompted); set `input_updater_capture_device = None` to use only the images in `input_images/`.

### Running the tests

    $ uv run pytest

The tests compare every blob of each downloaded model against OpenCV's independent Caffe importer, check backprop against finite differences and deconv against a hand-written Zeiler and Fergus pass, and run the toolbox UI headlessly. Tests for models whose weights have not been fetched are skipped.



# Troubleshooting

If you have any problems running the Deep Vis Toolbox, here are a few things to try:

 * If a network fails to load with `type "..." is not supported by torchcaffe`, your model uses a Caffe layer type that `torchcaffe` does not implement yet. Supported types are Convolution, InnerProduct, ReLU, Pooling (max/average/global), LRN, Dropout, Softmax, Concat, Split, Flatten, Sigmoid, TanH and Eltwise; new layers can be added in `torchcaffe/net.py`.
 * Settings files from older versions of the toolbox may define `caffevis_caffe_root`; it is no longer used and can be removed.
 * If the toolbox runs but the keys don't respond as expected, this may be because keys behave differently on different platforms. Run `uv run ./test_keys.py` to test behavior on your system.
 * If none of that helps, feel free to [submit an issue](https://github.com/yosinski/deep-visualization-toolbox/issues). I might have left out an important detail here or there :).



# Other ways of running the toolbox

These options were written for the original Caffe-based version of the toolbox:

 * John Moeller has put together a [Docker container for the toolbox](https://github.com/fishcorn/dvtb-container). This should even work on Windows! (confirmation needed)

 * If you're desperate, it's also possible to [run the toolbox on Amazon EC2](doc/deep-vis-on-aws.md), but display will be much slower and images can be loaded only from file (not from webcam).



