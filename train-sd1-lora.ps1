
param(
	[string]$BaseModel = '~\Downloads\ai\stable-diffusion\checkpoints\NovelAI.safetensors',
	[Parameter(Mandatory)]
	[string]$ConfigFile,
	[Parameter(Mandatory)]
	[string]$LoraName,
	[string]$MixedPrecision = 'bf16',
	[int]$MaxSteps = 2000,
	[float]$UnetLr = 1e-4,
	[float]$TextLr = 7e-5,
	[int]$SaveEveryN = 100,
	[int]$NetworkDim = 32,
	[int]$NetworkAlpha = 16
)

.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process 1 train_network.py `
	--pretrained_model_name_or_path=$BaseModel `
	--dataset_config=$ConfigFile `
	--output_dir=$LoraName  `
	--output_name=$LoraName `
	--save_model_as=safetensors `
	--prior_loss_weight=1.0 `
	--max_train_steps=$MaxSteps `
	--unet_lr=$UnetLr `
	--text_encoder_lr=$TextLr `
	--optimizer_type="AdamW8bit" `
	--xformers `
	--mixed_precision=$MixedPrecision `
	--gradient_checkpointing `
	--cache_latents `
	--save_every_n_epochs=$SaveEveryN `
	--network_module=networks.lora `
	--network_dim=$NetworkDim `
	--network_alpha=$NetworkAlpha `
	--lr_scheduler=cosine_with_restarts `
	--lr_scheduler_num_cycles=1 `
	--enable_bucket `
	--max_train_epochs=10 `
	--save_every_n_steps=$SaveEveryN `
	--flip_aug `
	--face_crop_aug_range="2.0,4.0"
