import math
import torch

def get_rays_from_camera(viewpoint_camera, device="cuda"):
    """
    Generates world-space ray origins and unit directions for every pixel
    in Kerbl's Camera object.
    returns:
        rays_o: (H * W, 3) camera center in world coordinates
        rays_d: (H * W, 3) unit ray direction per pixel in world coordinates
    """
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width

    # Focal lengths in pixels from field of view
    fx = W / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = H / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))

    # Pixel grid (u, v)
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Camera-space ray directions (same convention used by 3DGS: +x right, +y down, +z forward)
    dirs_cam = torch.stack(
        [(u - W * 0.5) / fx, (v - H * 0.5) / fy, torch.ones_like(u)], # centers pixel coordinates around (0, 0) and converts to real-world coordinates
        dim=-1,
    )  # (H, W, 3)

    # Rotate directions into world space using camera-to-world rotation
    c2w_rot = viewpoint_camera.world_view_transform[:3, :3].to(device)
    rays_d = dirs_cam @ c2w_rot.T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = viewpoint_camera.camera_center.to(device).expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def volume_render_rays(
    model,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float = 2.0, # uses tighter set (near, far) than Kerbl as starting point
    far: float = 6.0,
    num_samples: int = 64,
    white_background: bool = True,
):
    """
    Renders RGB colors for a batch of R rays by marching M depth slices
    and applying front-to-back alpha compositing.
    rays_o, rays_d: (R, 3)
    returns:
        rgb_map: (R, 3) predicted pixel colors
    """
    R = rays_o.shape[0]
    device = rays_o.device

    # 1. Sample M depth steps t_1, ..., t_M between near and far planes
    t_vals = torch.linspace(near, far, num_samples, device=device)  # (M,)
    delta_z = (far - near) / num_samples

    if model.training:
        t_vals = t_vals + (torch.rand(R, num_samples, device=device) - 0.5) * delta_z # adding small jitter to prevent overfitting
    else:
        t_vals = t_vals.unsqueeze(0).expand(R, num_samples)

    # 2. Compute 3D world coordinates pts = o + t * d -> shape (R, M, 3)
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)

    # 3. Query FactorizedGaussianModel -> sigma_total (R, M, 1), color_weighted (R, M, 3)
    sigma_total, color_weighted = model(pts)
    sigma = sigma_total.squeeze(-1)  # (R, M)

    # 4. Front-to-back volume rendering
    alpha = 1.0 - torch.exp(-sigma * delta_z)  # (R, M)

    ones = torch.ones((R, 1), device=device)
    transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]  # (R, M)

    weights = transmittance * alpha  # (R, M)
    rgb_map = torch.sum(weights.unsqueeze(-1) * color_weighted, dim=1)  # (R, 3)

    if white_background:
        acc_alpha = torch.sum(weights, dim=-1, keepdim=True)  # (R, 1)
        rgb_map = rgb_map + (1.0 - acc_alpha)

    return rgb_map

def render_factorized(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    num_samples: int = 64,
    ray_indices: torch.Tensor = None,
    chunk_size: int = 8192,
):
    """
    Factorized-Gaussian replacement for Kerbl's render() function.
    - If ray_indices is provided (training): renders only that random subset of rays.
    - If ray_indices is None (evaluation): renders the full image in VRAM-safe chunks.
    """
    device = pc._xy.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    white_bg = bool(bg_color.sum().item() > 1.5)

    rays_o, rays_d = get_rays_from_camera(viewpoint_camera, device=device)

    if ray_indices is not None: # training
        rgb = volume_render_rays(
            pc,
            rays_o[ray_indices],
            rays_d[ray_indices],
            num_samples=num_samples,
            white_background=white_bg,
        )
        return {"render": rgb}

    rgb_chunks = []
    for start in range(0, H * W, chunk_size): # evaluation
        end = min(start + chunk_size, H * W)
        chunk_rgb = volume_render_rays(
            pc,
            rays_o[start:end],
            rays_d[start:end],
            num_samples=num_samples,
            white_background=white_bg,
        )
        rgb_chunks.append(chunk_rgb)

    full_img = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
    return {"render": full_img}