"""Splatent model implementation."""
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from einops import rearrange, repeat


def make_1step_sched():
    """Create a 1-step DDPM scheduler for SD-Turbo."""
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler.set_timesteps(1, device="cuda")
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()
    return noise_scheduler


def load_ckpt_from_state_dict(net_splatent, optimizer, pretrained_path):
    """Load checkpoint from state dict."""
    sd = torch.load(pretrained_path, map_location="cpu")
    net_splatent.unet.load_state_dict(sd["state_dict_unet"], strict=False)
    optimizer.load_state_dict(sd["optimizer"])
    return net_splatent, optimizer


def save_ckpt(net_splatent, optimizer, outf):
    """Save checkpoint to file."""
    sd = {
        "state_dict_unet": net_splatent.unet.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(sd, outf)


class Splatent(torch.nn.Module):
    """Splatent: Reference-based diffusion enhancement for 3D Gaussian Splatting."""
    
    MODEL_REPO = "stabilityai/sd-turbo"
    
    def __init__(self, pretrained_path=None, timestep=999):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_REPO, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.MODEL_REPO, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(self.MODEL_REPO, subfolder="vae").cuda()
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.sched = make_1step_sched()
        
        self.unet = self._load_unet(pretrained_path)
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        
        self._print_model_info()
    
    def _load_unet(self, pretrained_path):
        """Load UNet model with optional pretrained weights."""
        config = UNet2DConditionModel.load_config(self.MODEL_REPO, subfolder="unet")
        unet = UNet2DConditionModel.from_config(config)
        
        model_file = hf_hub_download(self.MODEL_REPO, filename="unet/diffusion_pytorch_model.safetensors")
        state_dict = load_file(model_file)
        unet.load_state_dict(state_dict, strict=False)
        
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet.load_state_dict(sd["state_dict_unet"], strict=False)
        
        return unet.to("cuda")
    
    def _print_model_info(self):
        """Print model parameter information."""
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6
        print("=" * 50)
        print(f"Number of trainable parameters in UNet: {trainable_params:.2f}M")
        print("=" * 50)

    def set_eval(self):
        self.unet.eval()
        self.unet.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.unet.requires_grad_(True)

    def _encode_prompt(self, prompt, prompt_tokens):
        """Encode text prompt to embeddings."""
        if prompt is not None:
            caption_tokens = self.tokenizer(
                prompt, 
                max_length=self.tokenizer.model_max_length,
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.cuda()
            return self.text_encoder(caption_tokens)[0]
        return self.text_encoder(prompt_tokens)[0]
    
    def _arrange_grid(self, x):
        """Arrange multi-view features into spatial grid."""
        B, V, C, H, W = x.shape
        
        cols = (V + 1) // 2 if V > 1 else 1
        rows = (V + cols - 1) // cols
        if rows * cols != V:
            raise ValueError(f"Grid attention requires V that forms a rectangle, got V={V}")
        
        x = rearrange(x, 'b (r c) ch h w -> b ch (r h) (c w)', r=rows, c=cols).unsqueeze(1)
        return x, rows, cols
    
    def _extract_from_grid(self, output_feature, rows, cols):
        """Extract views from spatial grid."""
        return rearrange(
            output_feature[:, 0], 
            'b ch (r h) (c w) -> b (r c) ch h w', 
            r=rows, 
            c=cols
        )
    
    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        
        caption_enc = self._encode_prompt(prompt, prompt_tokens)
        x, rows, cols = self._arrange_grid(x)
        num_views = x.shape[1]
        
        z = rearrange(x, 'b v c h w -> (b v) c h w')
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        
        model_pred = self.unet(z, self.timesteps, encoder_hidden_states=caption_enc).sample
        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
        output_feature = rearrange(z_denoised, '(b v) c h w -> b v c h w', v=num_views)
        
        return self._extract_from_grid(output_feature, rows, cols)
    
    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        """Sample from the model given an input image and optional reference."""
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        transform = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        if ref_image is None:
            x = transform(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([transform(image), transform(ref_image)], dim=0).unsqueeze(0).cuda()
        
        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        return output_pil.resize((input_width, input_height), Image.LANCZOS)

    def save_model(self, outf, optimizer):
        """Save model checkpoint (LoRA and conv_in layers only)."""
        sd = {
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() 
                               if "lora" in k or "conv_in" in k},
            "optimizer": optimizer.state_dict()
        }
        torch.save(sd, outf)
