CUDA_VISIBLE_DEVICES=1 python train.py --model_name=UNet

CUDA_VISIBLE_DEVICES=1 python train.py --model_name=BSRoformer \
	--clip_duration=4.0 \
	--batch_size=4 \
	--lr=3e-4

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --model_name=BSRoformer4a --clip_duration=4.0 --batch_size=6 --lr=3e-4

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
	--model_name="BSRoformer" \
	--ckpt_path="checkpoints/train_accelerate/BSRoformer/step=390000.pth" \
	--clip_duration=2. \
	--batch_size=16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 train_accelerate.py --model_name=BSRoformer2 --clip_duration=2.0 --batch_size=3 --lr=3e-4


CUDA_VISIBLE_DEVICES=0 python inference.py \
	--model_name="BSRoformer" \
	--ckpt_path="checkpoints/train_accelerate/BSRoformer/step=390000.pth" \
	--clip_duration=2.0 \
	--batch_size=2


####
CUDA_VISIBLE_DEVICES=1 python train_enc_dec.py --model_name=EncDec --clip_duration=4.0 --batch_size=4 --lr=3e-4


# BSRoformer3	window=8192
# BSRoformer4a	mag, phase
# BSRoformer4b	dct
# BSRoformer5	fixed
# BSRoformer6a	full 2d 