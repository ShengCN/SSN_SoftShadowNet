#python render.py --gpus 0 --ds_root="Models/General" --out_folder="dataset/General" --start_id=0 --end_id=17
# python render.py --gpus 0 --ds_root="Models/Human" --out_folder="dataset/Human" --start_id=0 --end_id=12

ds_root="models"
cache="tmp/cache"
start_id=0
end_id=-1
width=512
height=512
cpus=32
out_hdf5="tmp/all_base.hdf5"

python render.py --cpus=$cpus --gpus 0 --ds_root=$ds_root --out_folder=$cache --start_id=$start_id --end_id=$end_id --width=$width --height=$height --cam_pitch 15 --model_rot -45 
python build_hdf5.py --cache=$cache --width=$width --height=$height --out_hdf5=$out_hdf5
