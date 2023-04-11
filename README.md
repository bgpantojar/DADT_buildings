# Damage augmented DT
This repository contains the codes for computing damage augmented digital twins as 3D models for buildings. The hereby methodology was presented in the paper ["Damage-augmented digital twins towards the automated inspection of buildings" by Pantoja-Rosero et., al. (2023)] (https://doi.org/10.1016/j.autcon.2023.104842)

<p align="center">
  <img src=docs/dadt_01.png>
</p>


<p align="center">
  <img src=docs/dadt_02.png>
</p>


## How to use it? (Note: tested for ubuntu 18.04lts)

### 1. Clone repository

Clone repository in your local machine. All codes related with method are inside the `src` directory.

### 2. Download data and CNN weights

Example input data can be downloaded from [Dataset for damage-augmented digital twins towards the automated inspection of buildings](https://doi.org/10.5281/zenodo.7767478). This datased contains 5 main folders. Images, SfM, weights, data and results. Extract the folders `data/` and `weights/` and place them inside the repository folder

#### 2a. Repository directory

The repository directory should look as:

```
DADT_buildings
└───data
└───docs
└───examples
└───src
└───weights
```

### 3. Environment

Create a conda environment and install python packages. At the terminal in the repository location.

`conda create -n DADT_buildings python=3.7`

`conda activate DADT_buildings`

`pip install -r requirements.txt`

`pip3 install torch torchvision`

### 4. Third party software

The method needs as input Structure from Motion information and LOD2 model that are computed by [Meshroom](https://github.com/alicevision/meshroom) and [Polyfit](https://github.com/LiangliangNan/PolyFit) respectively. Please refeer to the links to know how to use their methodologies.

In addition to create the final 3D DADT building models, it is necessary [FreeCAD](https://www.freecadweb.org/downloads.php) python console and its methods. You can either download the appimage and extract their content as `freecad_dev` or download the folder here [freecad_dev](https://drive.google.com/file/d/1LvjPHkhyo_gdBkCyHqN6uEqLqCGaB3vG/view?usp=sharing). Place the folder `freecad_dev` in the repository location. The repository directory should look as:

```
lod3_buildings
└───data
└───docs
└───examples
└───freecad_dev
  └───usr
    └───bin
    └───...
└───src
└───weights
```

### 5. Testing method with pusblished examples

Inside the folder `examples/` we have provide the input scripts that our algorithm needs. Two input scripts are necessary: `..._main_damage.py` and `..._main_DADT.py`. To run for instance the example `p4_DADT_00_La_capite` simply open the terminal inside the src folder (with the environment activated) and write the next command:

`./DADT.sh ../examples/p4_DADT_00_la_capite_main_damage.py ../examples/p4_DADT_00_la_capite_main_DADT.py`

The algorithm first will create the openings, damage and characterization in 3D and then merge them with the LOD2 model. Run the other examples similarly to the previous inline command.

`IMPORTANT` change the paths of the lines 3 and 5 in the file DADT.sh according your pc.

### 6. Creating your own digital twin as LOD3 model

Create a folder `your_DADT_data_folder` inside the `data\` folder. Inside `your_DADT_data_folder` create extra folders with the next structure:
```
lod3_buildings
└───data
  └───your_LOD3_data_folder
    └───images
      └───im1
    └───polyfit      
    └───sfm
    └───textured      
...
```

The methodology requires as input the next:

- sfm.json: file containing the sfm information (camera poses and structure). Add to the default `Meshroom` pipeline a node `ConverSfMFormat` and connect its input to the SfMData output from the node `StructureFromMotion`. In the node `ConverSfMFormat` options modify the SfM File Format to json. After running the modified `Meshroom` pipeline, this file is output in the folder `MeshroomCache/ConvertSfMFormat/a_folder_id/`. Copy that file inside the `your_DADT_data_folder\sfm`
- A registered view image for each facade containing the openings and damages: For each facade, place one image in which the openings and damages are visible in the folder `data/your_LOD3_data_folder/images/im1/`.
- polyfit.obj: use `Polyfit` pipeline either with the sparse or dense point cloud produced by the `Meshroom` pipeline. Note that it might be necessary to pre-process the point clouds deleting noisy points before running `Polyfit`. Save the output file as polyfit.obj or polyfit_dense.obj and place it in the folder `data/your_LOD3_data_folder/polyfit/`

Check the files of the data examples provided if neccessary to create the inpute data.

Finally create the two input scripts (`your_DADT_main_damage.py` and `your_DADT_main_DADT.py`) following the contents the given examples. Open the terminal inside the src folder (with the environment activated) and write the next command:

`./LOD3.sh ../examples/your_DADT_main_damage.py ../examples/your_DADT_main_DADT.py`


### 7. Results

The results will be saved inside `results` folder. Images of the pipeline stages are saved together with .obj files for the openings in 3D and the LOD3 model. Damage information is storaged as .ply files

#### 7.a Final repository directory

The repository directory after runing the medothology looks as:

```
lod3_buildings
└───data
└───docs
└───examples
└───freecad_dev
└───results
└───src
└───weights
```

### 8. Citation

We kindly ask you to cite us if you use this project, dataset or article as reference.

Paper:
```
@article{Pantoja-Rosero2023a,
title = {Damage-augmented digital twins towards the automated inspection of buildings},
journal = {Automation in Construction},
volume = {150},
pages = {104842},
year = {2023},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2023.104842},
url = {https://www.sciencedirect.com/science/article/pii/S0926580523001024},
author = {B.G. Pantoja-Rosero and R. Achanta and K. Beyer},
}
```
Dataset:
```
@dataset{Pantoja-Rosero2023a-ds,
  author       = {Pantoja-Rosero, Bryan German and
                  Achanta, Radhakrishna and
                  Beyer, Katrin},
  title        = {{Dataset for damage-augmented digital twins towards the automated inspection of buildings}},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v.0.0},
  doi          = {10.5281/zenodo.7767478},
  url          = {https://doi.org/10.5281/zenodo.7767478}
}
```
