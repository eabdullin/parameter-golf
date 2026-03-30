export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ENCODER_RATIO=0.2
export LATENT_LOSS_LAMBDA=0.2
export LATENT_PATCH_SIZE=4
export SIGREG_LAMBDA=0.01
export RUN_ID="lejepa_sp${VOCAB_SIZE}_sigreg${SIGREG_LAMBDA}_latloss${LATENT_LOSS_LAMBDA}_enc${ENCODER_RATIO}_latpatch${LATENT_PATCH_SIZE}"

torchrun --standalone --nproc_per_node=1 train_gpt.py