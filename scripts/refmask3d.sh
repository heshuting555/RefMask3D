###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################

#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# TRAIN and TEST
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python dist_train.py \
general.project_name="referring" \
general.experiment_name="RefMask3D_ckpt" \
general.eval_on_segments=true \
general.train_on_segments=true \
general.num_targets=2 \
general.topk_per_image=1 \
general.checkpoint='checkpoints/scannet/scannet_val.ckpt' \
general.gpus=8 \
trainer.check_val_every_n_epoch=1 \
trainer.max_epochs=20 \
trainer.num_sanity_val_steps=8 \
model=refmask3d \
model.num_queries=100 \
model.config.freeze_text_encoder=false \
loss=set_criterion_contrastive \
data.batch_size=4 \
optimizer.lr=6e-5

