import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        
        # 添加固定的LayerNorm层
        #self.encoder_layer_norm = nn.LayerNorm(hid_dim)
        # 不训练这个LayerNorm的参数，保持随机初始化状态
        #for param in self.encoder_layer_norm.parameters():
        #    param.requires_grad = False

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        #x = self.encoder_layer_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        original_slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = original_slots.reshape((-1, original_slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        # change the slots to original_slots
        return recon_combined, recons, masks, original_slots


from dataclasses import dataclass

@dataclass
class SlotAttentionConfig:
    num_slots: int = 7
    num_iterations: int = 3
    hid_dim: int = 64
    pretrained_path: str = None
    freeze_encoder: bool = True
    use_slots_as_patches: bool = True


class SlotAttentionEncoder(nn.Module):
    def __init__(
        self,
        obs_shape:List[int],
        cfg:SlotAttentionConfig,
        num_channel=3,
        img_h=128,
        img_w=128,
        #skip_decoder=True,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.img_h = img_h
        self.img_w = img_w

        resolution = (img_h, img_w)
        self.slot_attention_model = SlotAttentionAutoEncoder(
            resolution=resolution,
            num_slots=cfg.num_slots,
            num_iterations=cfg.num_iterations,
            hid_dim=cfg.hid_dim,
            #skip_decoder=skip_decoder,
        )
        '''
        self.slot_attention_model.load_state_dict(torch.load(cfg.pretrained_path))
        if cfg.freeze_encoder:
            for param in self.slot_attention_model.encoder_cnn.parameters():
                param.requires_grad = False
            print("Freezing encoder parameters")


        '''
        '''
        if cfg.pretrained_path:
            try:
                checkpoint = torch.load(cfg.pretrained_path, map_location = 'cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                self.slot_attention_model.load_state_dict(state_dict)
                print("successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        '''
        if cfg.pretrained_path:
            try:
                checkpoint = torch.load(cfg.pretrained_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 使用strict=False允许部分加载
                missing_keys, unexpected_keys = self.slot_attention_model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys (will be randomly initialized): {missing_keys}")
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        if cfg.freeze_encoder:
            #for param in self.slot_attention_model.encoder_cnn.parameters():
            #    param.requires_grad = False
            #print("Freezing encoder parameters")
            for param in self.slot_attention_model.parameters():
                param.requires_grad = False
            print("Freezing slot attention model parameters")
        '''
        if cfg.use_slots_as_patches:
            # Treat each slot as a patch
            self.num_patch = cfg.num_slots
            self.patch_repr_dim = cfg.hid_dim
            self.repr_dim = cfg.hid_dim * cfg.num_slots
        else:
            #use the fixed number of patches
            self.num_patch = cfg.num_slots
            self.patch_repr_dim = cfg.hid_dim
            self.repr_dim = cfg.hid_dim * cfg.num_slots
        '''
        self.num_slots = cfg.num_slots
        self.slot_dim = cfg.hid_dim
        self.total_slot_dim = cfg.hid_dim * cfg.num_slots

    def load_pretrained_weights(self, path):
        checkpoint = torch.load(path, map_location = 'cpu')
        self.slot_attention_model.load_state_dict(checkpoint)

    def forward(self, obs, flatten=False):
        '''
        Forward pass that extracts slot representations

        Args:
            obs: Input image tensor of shape (B, C, H, W)
        Returns:
            slot_representations: Tensor of shape (B, num_slots, repr_dim)
        '''
        obs = obs / 255.0
        #if self.slot_attention_model.skip_decoder:
        # skip decoder
        x = self.slot_attention_model.encoder_cnn(obs)
        #x = self.layer_norm(x)
        #x = self.slot_attention_model.encoder_layer_norm(x)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.slot_attention_model.fc1(x)
        x = F.relu(x)
        x = self.slot_attention_model.fc2(x)
        slots = self.slot_attention_model.slot_attention(x)
        #else:
        #_, _, _, slots = self.slot_attention_model(obs)

        if flatten:
            slots = slots.flatten(1, 2) # (B, num_slots* hidden_dim)
        return slots
