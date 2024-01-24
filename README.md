# rPPG pipeline
rPPG pipeline with segment anything implementation

## Prerequisites
- Python 3.7 or higher
- Pip package manager
- Poetry 

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/rppg_pipeline.git
    cd rppg_pipeline
    ```

2. Download the model weights:
    - [Haarcascade](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml)
    - [Segment Anything](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    - [LBFmodel](https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml)

3. Install the required dependencies:
    ```bash
    poetry install 
    ```

## Local Usage
1. Run the pipeline:
    ```bash
    poetry run python rppg/main.py
    ```

## Google Cloud (gcloud) Usage
To deploy the rPPG pipeline on Google Cloud, follow these steps:

1. Set up a Google Cloud project and enable the necessary APIs.

2. Create a virtual machine instance on Google Compute Engine.

3. SSH into the virtual machine and clone the repository:
    ```bash
    git clone https://github.com/your-username/rppg_pipeline.git
    cd rppg_pipeline
    ```

4. Download the model weights:
    - [Haarcascade](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml)
    - [Segment Anything](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    - [LBFmodel](https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml)

5. Install the required dependencies:
    ```bash
    poetry install 
    ```

6. Upload the data to a Google Cloud Storage bucket.

7. Run the pipeline:
    ```bash
    poetry run python rppg/main.py
    ```

## Note 
The parameters in the code are set to use the base model, which means that memory usage is not a limitation and the pipeline can be run locally. However, if you want to improve performance, you can adjust the model weights size and frequency accordingly.

## Contributing
Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md) when making changes to the project.

## License
This project is licensed under the [MIT License](LICENSE).
