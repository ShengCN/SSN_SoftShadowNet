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
samples=1
base_samples=2

python render.py --cpus=$cpus --gpus 0 --model_root=$model_root --out_folder=$cache --start_id=$start_id --end_id=$end_id --width=$width --height=$height --samples=$samples --cam_pitch_min=$cam_pitch_min --cam_pitch_max=$cam_pitch_max --model_rot_min=$model_rot_min --model_rot_max=$model_rot_max --base_samples=$base_samples &&

rm tmp/scene_cache.bin
python build_hdf5.py --cache=$cache --width=$width --height=$height --out_hdf5=$out_hdf5


model_root="models"
cache="Dataset/human_data/tmp/cache"
start_id=0
end_id=-1 # -1 means all models
width=512
height=512
cpus=6
out_hdf5="Dataset/human_data/all_base2.hdf5"
cam_pitch_min=0
cam_pitch_max=45
model_rot_min=-90
model_rot_max=90
samples=1
patch_size=8

python render.py --cpus=$cpus --gpus 0 --model_root=$model_root --out_folder=$cache --start_id=$start_id --end_id=$end_id --width=$width --height=$height --samples=$samples --cam_pitch_min=$cam_pitch_min --cam_pitch_max=$cam_pitch_max --model_rot_min=$model_rot_min --model_rot_max=$model_rot_max --patch_size=$patch_size &&

rm tmp/scene_cache.bin
python build_hdf5.py --cache=$cache --width=$width --height=$height --out_hdf5=$out_hdf5
