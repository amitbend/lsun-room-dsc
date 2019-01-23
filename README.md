# Indoor Layout Estimation from a Single Image

![one_lsun_result_banner](./doc/banner.png)

## TODO
- Upload trained model.
- Upload test results.

## [Original readme](https://github.com/leVirve/lsun-room)
## [Original paper]()
## Usage
- Install libraries according to [requirements](https://github.com/shuuchen/lsun-room-dsc/blob/master/requirements.txt).
- Download dataset from http://lsun.cs.princeton.edu/2015.html#layout and put them in following folders.
- Datasets preparation

  - Put `LSUN Room Layout Dataset` in folder `../data/lsun_room` relative to this project.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene
  - Run the following to prepare train/evaluation datasets.
    ```bash
    python re_label.py
    ```
  
- Training
  - The trained model will be saved to folder ./exp/checkpoints/
  - You can modify config.yml to play with hyperparameters for training.
  
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
  - --input_path/--output_path point to the folders of input/output images.
  
  ```bash
  python demo.py --weight [weight_path] --input_path [input_image_dir] --output_path [output_image_dir]

  Usage: demo.py [OPTIONS]

  Options:
    --input_path PATH
    --output_path PATH
    --weight PATH
    --input_size <INTEGER INTEGER>...
    --help                          Show this message and exit.

  ```

MIT License
