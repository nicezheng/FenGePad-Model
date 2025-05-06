# Official model implementation of [FenGePad--A Tangible Multi-Prompt Interactive Framework for Deep Dune Segmentation]()
## Overview

**FenGePad** is a tablet-based interactive segmentation tool that enables users to segment *unseen* structures in dune images using clicks, polylines and scribbles. This project details segmentation model of FenGePad.

## Training Dune Dataset
Dune segmentation dataset comes from high-resolution remote sensing imagery captured in April 2020 by the \textit{Pleiades} satellite over the Mu Us Sandy Land, a major arid region in Northwestern China. The 0.5-meter images were orthorectified and geometrically corrected, with control points distributed throughout the landscape and densified around dune structures. Expert geographers manually annotated dune boundaries through visual interpretation in ArcGIS. The final dataset comprises 1,537 training samples from the core dune areas and 174 test samples from an adjacent area, facilitating robust generalization evaluation.

## Installation

You can install `FenGePad` by cloning it and installing dependencies:
```
git clone https://github.com/nicezheng/FenGePad-Model
python -m pip install -r ./FenGePad/requirements.txt
python -m pip install -r ./FenGePad/requirements_training.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./FenGePad)"
```

## Getting Started
To instantiate FenGePad-UNet and make a prediction:
```
from fengepad import FenGePadUNet

fgp_unet = FenGePadUNet()

mask = fgp_unet.predict(
    image,        # (B, 1, H, W) 
    point_coords, # (B, n, 2)
    point_labels, # (B, n)
    scribbles,    # (B, 2, H, W)
    box,          # (B, n, 4)
    mask_input,   # (B, 1, H, W)
) # -> (B, 1, H, W) 
```

For best results, `image` should have spatial dimensions $(H,W) = (128,128)$ and pixel values min-max normalized to the $[0,1]$ range. `mask_input` should be the logits from the previous prediction.

## Training

>Note: our training code requires the [pylot](https://github.com/JJGO/pylot) library. The inference code above does not.  We recommend installing via pip:
>```
>pip install --timeout=60 git+https://github.com/JJGO/pylot.git@87191921033c4391546fd88c5f963ccab7597995
>```

The configuration settings for training are controlled by yaml config files. We provide an example configs in [`./configs`](https://github.com/nicezheng/FenGePad-Model/configs) for training from scratch on an example dataset.

To train a model from scratch:
```
python fengepad/experiment/unet.py -config train_dune_unet.yaml 
```


## Acknowledgements

* Our training code builds on the [`pylot`](https://github.com/JJGO/pylot) library for deep learning experiment management. Thanks to [@JJGO](https://github.com/JJGO) for sharing this code! 

* We use functions from [voxsynth](https://github.com/dalcalab/voxynth) for applying random deformations during scribble simulation 

* Our project is developed based on [Scribbleprompt](https://github.com/halleewong/ScribblePrompt/). Thanks for the nice demo GUI and code architecture:)


## License

Code for this project is released under the Apache 2.0 License.




