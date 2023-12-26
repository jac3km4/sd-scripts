
param(
	[string]$BaseModel = '~\Downloads\ai\stable-diffusion\checkpoints\v1.5.safetensors',
	[Parameter(Mandatory)]
	[string]$ConfigFile,
	[Parameter(Mandatory)]
	[string]$TiName,
	[Parameter(Mandatory)]
	[string]$InitWord,
	[Parameter(Mandatory)]
	[int]$NumVectors,
	[string]$MixedPrecision = 'bf16',
	[int]$MaxSteps = 1600,
	[float]$LeaningRate = 14e-4,
	[int]$SaveEveryN = 100
)

.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py `
	--pretrained_model_name_or_path=$BaseModel `
	--dataset_config=$ConfigFile `
	--output_dir=$TiName  `
	--output_name=$TiName `
	--save_model_as=safetensors `
	--prior_loss_weight=1.0 `
	--max_train_steps=$MaxSteps `
	--optimizer_type="AdamW8bit" `
	--xformers `
	--mixed_precision=$MixedPrecision `
	--cache_latents `
	--gradient_checkpointing `
	--learning_rate=$LeaningRate `
	--lr_scheduler=cosine_with_restarts `
	--lr_scheduler_num_cycles=1 `
	--enable_bucket `
	--flip_aug `
	--face_crop_aug_range="2.0,4.0" `
	--token_string=$TiName `
	--num_vectors_per_token=$NumVectors `
	--save_every_n_steps=$SaveEveryN
