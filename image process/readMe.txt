1. To rename images file name, use:
python rename_all_images.py -ip "C:\Users\81955\Desktop\Expression_Transformation\datasets\face\Original_Img" -si 0

2. To [detect and crop face] & [generate face bound file(json)]:
jupiter use: FaceProcess.ipynb -> GetFaceBoundBox
ps：因为window不支持face_recognition，只能在colab上运行
others use: get_face_and_bound.py


3. To calculate AUs for each image you use:
cd dir<./OpenFace_2.2.0_win_x86/> at cmd
FaceLandmarkImg.exe -fdir "C:\Users\81955\Desktop\datasets\face\imgs" -aus
ps: then result will be stored in  ./OpenFace_2.2.0_win_x86/processed/


4. To generate the aus_openface.pkl, use:
python prepare_au_annotations.py -ia "C:\Users\81955\Desktop\OpenFace_2.2.0_win_x86\processed" -op "C:\Users\81955\Desktop\datasets\face"
ps: You should first extract each image Action Units with OpenFace and store each output in a csv file the same name as the image. -> step3


5. To make csv ID file for train or test, use:
python make_ID_file.py -ip "C:\Users\81955\Desktop\datasets\face\imgs" -op "C:\Users\81955\Desktop\datasets\face" -m test
ps: if can do it yourself :)


6. To test, you can define you own test mode:
# random target mode: select a target image randomly for every image
python main.py --mode=test --data_root=./datasets/face --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\Expression_Transformation\ckpts\face\ganimation\200411_174647
python main.py  --mode=test --data_root=./datasets/face --interpolate_len=8 --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\Expression_Transformation\ckpts\face\ganimation\200411_174647
python main.py --mode=test --data_root=./datasets/face --interpolate_len=8 --load_size=200 --final_size=200 --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\Expression_Transformation\ckpts\face\ganimation\200411_174647
python main.py  --mode=test --data_root=C:\Users\81955\Desktop\datasets\face --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\ckpts\face\ganimation\200411_174647 --test_mode=single_target --single_target_img=C:\Users\81955\Desktop\datasets\face\imgs\63.jpg


# single target mode: a single target for all test images
python main.py --mode=test --data_root=./datasets/face --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\Expression_Transformation\ckpts\face\ganimation\200411_174647 --test_mode=single_target --single_target_img=C:\Users\81955\Desktop\Expression_Transformation\datasets\face\imgs\117.jpg --save_all_alpha_image
python main.py --mode=test --data_root=./datasets/face --interpolate_len=6 --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\Expression_Transformation\ckpts\face\ganimation\200411_174647 --test_mode=single_target --single_target_img=C:\Users\81955\Desktop\Expression_Transformation\datasets\face\imgs\19_0.jpg --save_all_alpha_image
python main.py --mode=test --data_root=C:\Users\81955\Desktop\datasets\face --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\ckpts\face\ganimation --test_mode=single_target --single_target_img=C:\Users\81955\Desktop\datasets\face\imgs\117.jpg


# if your dataset has many image with different size, you should use resize_or_crop:
python main.py --mode=test --data_root=C:\Users\81955\Desktop\datasets\face --batch_size=1 --n_threads=0 --load_epoch=30 --ckpt_dir=C:\Users\81955\Desktop\ckpts\face\ganimation\200411_174647 --test_mode=single_target --single_target_img=C:\Users\81955\Desktop\datasets\face\imgs\0.jpg --resize_or_crop=resize_and_crop --load_size=200 --final_size=200
# pair target mode: offer a pair of <original_image, target_image>
# todo


7. To split join face:
jupiter use: FaceProcess.ipynb::FaceSplitJoin
others use: FaceSplitJoin.py.


8. To beautify face:
# todo
face_beautify.py
ps: For erase black texture in some generate images.


9 To [train] or [finetune]，Note: --load_epoch为加载的预训练模型的起始epoch，--epoch_count为开始训练时的epoch， 总epoch（包括预训练模型的epoch） = niter + niter_decay
即训练epoch从epoch_count到niter+niter_dcay

finetune:
python main.py --data_root datasets/celebA --sample_img_freq 300 --n_threads 18  --ckpt_dir ckpts/face/ganimation/200411_174647 --load_epoch 30 --epoch_count 31 --niter 30 --niter_decay 30 --save_epoch_freq 5

python main.py --data_root ../celebA --n_threads 16 --ckpt_dir ckpts/celebA/ganimation/200510_180637 --load_epoch 30 --epoch_count 31 --niter 30 --niter_decay 10

python main.py --mode=test --data_root=./datasets/face --interpolate_len=5 --batch_size=25 --n_threads=16 --load_epoch=45 --ckpt_dir=ckpts/face/ganimation/200411_174647

python main.py --mode=test --data_root=./datasets/face --n_threads=18 --load_epoch=30 --ckpt_dir=ckpts/face/ganimation/200411_174647 --test_mode=single_target --single_target_img=datasets/face/imgs/117.jpg

train:
python main.py --data_root datasets/celebA --display


C:\Users\81955\Desktop\ckpts\face\ganimation\200411_174647