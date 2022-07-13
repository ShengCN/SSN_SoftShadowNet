# SSN: Soft Shadow Network for Image Composition
Remember to recursively git clone this repository. 
```Bash
git clone --recurse-submodules -j8 https://github.com/ShengCN/SSN_SoftShadowNet.git
```

## Data Render
The renderer for the data is in `data` folder. This renderer is accelerated by CUDA. The computation time will be proportional to final image dimension and triangle numbers. 

There is an example testing model: `data/test_model.off` which has **19317** triangles. For the **512x512** output image, a mask rendering takes 0.018 s and a **8x8** shadow base takes 3.6s for a GTX Titan Xp. 

### Environment Requirements
* System: Ubuntu 18.04
* CMake: >= 3.8
* GCC: >= 7.4.0
* CUDA (Ensure CMake can find nvcc)

CMake is used to manage the packages and compiling. All other dependencies required except CUDA are in the `data/src/Dep` folders. Again, remember to recursively clone the submodules. 

There is a pre-compiled Ubuntu(18.04) version in `data/src/build` folder. If you cannot run the `data/src/build/run.sh` script, then you need to recompile. 

### (Optional) Recompile 
```Bash
cd data/src/build
rm -r CMake*
cmake ..
make -j16
```

### Command Interface
1. Change directory to Data/src
2. Put obj/off files to models folder 
3. Check the parameters in the render.sh, e.g. gpus(support multiple gpu), random pitch range, random model rotation?

``` Bash
# File render.sh
model_root="models"
cache="Dataset/human_data/cache"
start_id=0
end_id=-1 # -1 means all models
width=512
height=512
cpus=6
out_hdf5="Dataset/human_data/all_base.hdf5"
cam_pitch_min=0
cam_pitch_max=45
model_rot_min=-90
model_rot_max=90
samples=2

rm tmp/scene_cache.bin

python render.py --cpus=$cpus --gpus 0 2 --model_root=$model_root --out_folder=$cache --start_id=$start_id --end_id=$end_id --width=$width --height=$height --samples=$samples --cam_pitch_min=$cam_pitch_min --cam_pitch_max=$cam_pitch_max --model_rot_min=$model_rot_min --model_rot_max=$model_rot_max &&
python build_hdf5.py --cache=$cache --width=$width --height=$height --out_hdf5=$out_hdf5
```
4. Run ./render.sh




# License
SSN may be used non-commercially, meaning for research or evaluation purposes only. 

# Citation
```
@inproceedings{sheng2021ssn,
  title={SSN: Soft shadow network for image compositing},
  author={Sheng, Yichen and Zhang, Jianming and Benes, Bedrich},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4380--4390},
  year={2021}
}
```
