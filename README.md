# Indoor Layout Estimation from a Single Image

![one_lsun_result_banner](./doc/banner.png)

## [Original readme](https://github.com/leVirve/lsun-room)
## [Original paper]()
## Usage
- Download dataset from http://lsun.cs.princeton.edu/2015.html#layout and put them in following folders.
- Dataset

  - Put `LSUN Room Layout Dataset` in folder `../data/lsun_room` relative to this project.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene
  - Run the following to prepare train/evaluation data.
    ```bash
    python re_label.py
    ```
  
- Training
  - The trained model will be saved to folder ./exp/checkpoints/
  
  ```bash
  python main.py --phase train --name train

  Usage: main.py [OPTIONS]

  Options:
    --name TEXT
    --dataset [lsun_room | others]
    --dataset_root TEXT
    --log_dir TEXT
    --image_size <INTEGER INTEGER>
    --epochs INTEGER
    --batch_size INTEGER
    --workers INTEGER
    --l1_weight FLOAT
    --resume PATH
  ```

- Prediction
  - Specify the weight path of a trained model.
  - The weight path should be a file named as net-xx.pt
  ```bash
  python demo.py --weight [weight_path]

  Usage: demo.py [OPTIONS]

  Options:
    --device INTEGER
    --video TEXT
    --weight TEXT
    --input_size <INTEGER INTEGER>.

  ```

MIT License
