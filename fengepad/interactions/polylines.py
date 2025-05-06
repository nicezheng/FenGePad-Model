
from typing import Union, Tuple, List, Optional
import scipy.ndimage as ndi
import numpy as np
import torch
import kornia
import cv2
import voxynth
import skimage.morphology
import skimage.measure
from .utils import _as_single_val
import random
import os
import skimage.graph
# Prevent neurite from trying to load tensorflow
os.environ['NEURITE_BACKEND'] = 'pytorch' 


# -----------------------------------------------------------------------------
# Parent class
# -----------------------------------------------------------------------------

class WarpScribble:
    """
    Parent scribble class with shared functions for generating noise masks (useful for breaking up scribbles) and applying deformation fields (to warp scribbles)
    """
    def __init__(self, 
                warp: bool = True,
                warp_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                warp_magnitude: Union[int,Tuple[int],List[int]] = (1, 6), 
                ):
        if isinstance(warp_smoothing, int):
            warp_smoothing = [warp_smoothing, warp_smoothing]
        if isinstance(warp_magnitude, int):
            warp_magnitude = [warp_magnitude, warp_magnitude]
        # Warp settings
        self.warp = warp
        self.warp_smoothing = list(warp_smoothing)
        self.warp_magnitude = list(warp_magnitude)
        # Noise mask settings
        # self.mask_smoothing = mask_smoothing
        self.mask_smoothing = (4, 8)
        
        
    def noise_mask(self, mask, shape: Union[Tuple[int],List[int]] = (8,128,128), device = None):
        """
        Get a random binary mask by thresholding smoothed noise. The mask is used to break up the scribbles
        """
        if isinstance(self.mask_smoothing, tuple):
            get_smoothing = lambda: np.random.uniform(*self.mask_smoothing)
        else:
            get_smoothing = lambda: self.mask_smoothing

        # noise = torch.stack([
        #     voxynth.noise.perlin(shape=shape[-2:], smoothing=get_smoothing(), magnitude=1, device=device) for _ in range(shape[0])
        # ]) # shape: b x H x W
        # noise_mask = (noise > 0).int().unsqueeze(1)

        # --- Compute distance transform ---
        mask_np = mask.squeeze(1).cpu().numpy()  # (B, H, W)
        dist_maps = []
        for i in range(mask_np.shape[0]):
            dist = ndi.distance_transform_edt(mask_np[i])
            dist_maps.append(dist)
        dist_map = torch.from_numpy(np.stack(dist_maps)).unsqueeze(1).float().to(mask.device)  # (B,1,H,W)

        # --- Generate perlin noise ---
        noise = torch.stack([
            voxynth.noise.perlin(shape=shape[-2:], smoothing=get_smoothing(), magnitude=1, device=device)
            for _ in range(shape[0])
        ]).unsqueeze(1)  # (B,1,H,W)

        # --- Adjust threshold based on distance ---
        # Larger distance -> harder to break; Smaller distance -> easier to break
        dynamic_noise = noise + dist_map * 0.7

        # Step 4: determine a threshold so that roughly target_keep_ratio remains
        flat_dynamic_noise = dynamic_noise.view(shape[0], -1)
        thresholds = []
        target_keep_ratio = 0.85
        for i in range(shape[0]):
            sorted_vals, _ = flat_dynamic_noise[i].sort(descending=True)
            cutoff_idx = int(target_keep_ratio * sorted_vals.size(0))
            threshold_val = sorted_vals[cutoff_idx]
            thresholds.append(threshold_val)

        thresholds = torch.tensor(thresholds, device=device).view(shape[0],1,1,1)
        noise_mask = (dynamic_noise > thresholds).int()

        return noise_mask # shaoe: b x 1 x H x W
    
    def apply_warp(self, x: torch.Tensor):
        """
        Warp a given mask x using a random deformation field
        """
        if x.sum() > 0:
            # warp scribbles using a deformation field
            deformation_field = voxynth.transform.random_transform(
                shape = x.shape[-2:],
                affine_probability = 0.0,
                warp_probability = 1.0,
                warp_integrations = 0,
                warp_smoothing_range = self.warp_smoothing,
                warp_magnitude_range = self.warp_magnitude,
                voxsize = 1,
                device = x.device,
                isdisp = False
                )

            warped = voxynth.transform.spatial_transform(x, trf = deformation_field, isdisp=False) 
            if warped.sum() == 0:
                return x
            else:
                return (warped - warped.min()) / (warped.max() - warped.min())
        else:
            # Don't need to warp if mask is empty
            return x
    
    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1):
        """
        Simulate scribbles for a batch of examples (mask).
        """
        raise NotImplementedError
    
    def __call__(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (b,1,H,W) or (1,H,W) mask in [0,1] to sample scribbles from
        Returns:
            scribble_mask: (b,1,H,W) or (1,H,W) mask(s) of scribbles on [0,1]
        """
        assert len(mask.shape) in [3,4], f"mask must be b x 1 x h x w or 1 x h x w. currently {mask.shape}"
        
        if len(mask.shape)==3:
            # shape: 1 x h x w
            return self.batch_scribble(mask[None,...], n_scribbles=n_scribbles)[0,...]
        else:
            # shape: b x 1 x h x w
            return self.batch_scribble(mask, n_scribbles=n_scribbles)


# -----------------------------------------------------------------------------
# Contour Polyline for Dune Segmentation (final corrected version)
# -----------------------------------------------------------------------------

# 继承你的 WarpScribble
class ContourPolylineForDune(WarpScribble):
    def __init__(self,
                 length_factor: float = 20.0,   # 控制弧长影响
                 curvature_factor: float = 100.0, # 控制弯曲度影响
                 min_points: int = 3,
                 max_points: int = 15,
                 close_polyline: bool = False,
                 show: bool = False):
        super().__init__(warp=False)
        self.length_factor = length_factor
        self.curvature_factor = curvature_factor
        self.min_points = min_points
        self.max_points = max_points
        self.close_polyline = close_polyline
        self.show = show

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: Optional[int] = 1):
        assert len(mask.shape) == 4, f"mask must be (b,1,h,w). Got {mask.shape}"
        bs = mask.shape[0]
        device = mask.device
        output = torch.zeros_like(mask)
        debug_contours = []
        debug_sampled_points = []

        for i in range(bs):
            mask_np = (mask[i, 0] > 0).cpu().numpy().astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask_np, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                debug_contours.append(torch.zeros_like(mask[i,0]))
                debug_sampled_points.append(torch.zeros_like(mask[i,0]))
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) < 2:
                debug_contours.append(torch.zeros_like(mask[i,0]))
                debug_sampled_points.append(torch.zeros_like(mask[i,0]))
                continue

            coords = largest_contour.squeeze(1)  # (N,2)

            # Compute arc length
            deltas = np.diff(coords, axis=0)
            dists = np.hypot(deltas[:,0], deltas[:,1])
            arc_length = np.sum(dists)

            # Compute total curvature (turning angles)
            vectors = deltas / (dists[:,None] + 1e-8)
            angles = np.arctan2(vectors[:,1], vectors[:,0])
            angle_diffs = np.diff(angles)
            angle_diffs = (angle_diffs + np.pi) % (2*np.pi) - np.pi  # wrap to [-pi, pi]
            total_curvature = np.sum(np.abs(np.degrees(angle_diffs)))

            # Determine number of points
            n_points = int(arc_length / self.length_factor + total_curvature / self.curvature_factor)
            n_points = np.clip(n_points, self.min_points, self.max_points)

            # Sample points uniformly by cumulative arc length
            cumulative = np.concatenate([[0], np.cumsum(dists)])
            sample_locs = np.linspace(0, cumulative[-1], n_points)
            sampled_coords = []
            for loc in sample_locs:
                idx = np.searchsorted(cumulative, loc)
                idx = min(idx, len(coords)-1)
                sampled_coords.append(coords[idx])
            sampled_coords = np.array(sampled_coords, dtype=np.int32)

            # Draw debug sampled points
            sampled_canvas = np.zeros_like(mask_np)
            for pt in sampled_coords:
                cv2.circle(sampled_canvas, tuple(pt), radius=1, color=255, thickness=-1)

            # Draw final polyline
            canvas = np.zeros_like(mask_np)
            cv2.polylines(canvas, [sampled_coords.reshape(-1,1,2)], isClosed=self.close_polyline, color=255, thickness=1)

            output[i,0] = torch.from_numpy(canvas / 255.0).float().to(device)
            debug_contours.append(torch.from_numpy(mask_np).float().to(device))
            debug_sampled_points.append(torch.from_numpy(sampled_canvas / 255.0).float().to(device))

        if self.show:
            self.visualize_batch(mask, output, debug_contours, debug_sampled_points)

        return output

    def visualize_batch(self, mask: torch.Tensor, output: torch.Tensor, debug_contours, debug_sampled_points):
        import matplotlib.pyplot as plt
        import neurite as ne
        from fengepad.analysis.plot import show_scribbles
        import numpy as np

        bs = mask.shape[0]
        tensors = []
        titles = []
        for i in range(bs):
            tensors += [
                mask[i,0].cpu(),
                debug_contours[i].cpu(),
                debug_sampled_points[i].cpu(),
                output[i,0].cpu()
            ]
            titles += ["Input Mask", "Contour", "Sampled Points", "Generated Polyline"]

        fig, axes = ne.plot.slices(
            tensors,
            titles,
            show=False,
            grid=(bs,4),
            width=4*3
        )
        axes = np.array(axes).reshape(-1)
        for i in range(bs):
            show_scribbles(output[i,0].detach().cpu(), axes[i*4 + 3])

        plt.savefig("contour_polyline_prompt_fixed.png")
        plt.show()
        

# -----------------------------------------------------------------------------
# Centerline Polyline for Dune Segmentation (Final Version)
# -----------------------------------------------------------------------------

class CenterlinePolylineForDune(WarpScribble):
    """
    Generate sparse polylines along dune centerlines:
    1) Extract medial skeleton
    2) Select connected centerline
    3) Uniformly sample points along cumulative length
    4) Connect sampled points into a polyline
    """
    def __init__(self,
                 sample_points_range=(3, 10),
                 min_skeleton_length=20,
                 close_polyline: bool = False,
                 show: bool = True):
        super().__init__(warp=False)
        self.sample_points_range = sample_points_range
        self.min_skeleton_length = min_skeleton_length
        self.close_polyline = close_polyline
        self.show = show

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: Optional[int] = 1):
        assert len(mask.shape) == 4, f"mask must be (b,1,h,w). Got {mask.shape}"
        bs = mask.shape[0]
        device = mask.device
        output = torch.zeros_like(mask)
        debug_skeletons = []
        debug_sampled_points = []

        for i in range(bs):
            mask_np = (mask[i, 0] > 0).cpu().numpy().astype(np.uint8)

            # Step 1: Skeletonization
            skeleton = cv2.ximgproc.thinning(mask_np)
            if np.count_nonzero(skeleton) < self.min_skeleton_length:
                debug_skeletons.append(torch.zeros_like(mask[i,0]))
                debug_sampled_points.append(torch.zeros_like(mask[i,0]))
                continue

            coords = np.column_stack(np.nonzero(skeleton))
            if len(coords) < 2:
                debug_skeletons.append(torch.zeros_like(mask[i,0]))
                debug_sampled_points.append(torch.zeros_like(mask[i,0]))
                continue

            # Step 2: Cumulative arc length
            coords = coords.astype(np.float32)
            deltas = np.diff(coords, axis=0)
            dists = np.hypot(deltas[:,0], deltas[:,1])
            arc_length = np.concatenate([[0], np.cumsum(dists)])

            n_points = random.randint(*self.sample_points_range)
            if len(arc_length) <= n_points:
                sampled_coords = coords
            else:
                sample_locs = np.linspace(0, arc_length[-1], n_points)
                sampled_coords = []
                for loc in sample_locs:
                    idx = np.searchsorted(arc_length, loc)
                    idx = min(idx, len(coords)-1)
                    sampled_coords.append(coords[idx])
                sampled_coords = np.array(sampled_coords)

            sampled_coords = np.array(sampled_coords, dtype=np.int32)

            if len(sampled_coords) < 2:
                debug_skeletons.append(torch.zeros_like(mask[i,0]))
                debug_sampled_points.append(torch.zeros_like(mask[i,0]))
                continue

            # Step 3: Draw sampled points
            sampled_canvas = np.zeros_like(mask_np)
            for pt in sampled_coords:
                cv2.circle(sampled_canvas, tuple(pt[::-1]), radius=1, color=255, thickness=-1)

            # Step 4: Draw polyline
            canvas = np.zeros_like(mask_np)
            cv2.polylines(canvas, [sampled_coords.reshape(-1,1,2)], isClosed=self.close_polyline, color=255, thickness=1)

            output[i,0] = torch.from_numpy(canvas / 255.0).float().to(device)
            debug_skeletons.append(torch.from_numpy(skeleton/255.0).float().to(device))
            debug_sampled_points.append(torch.from_numpy(sampled_canvas / 255.0).float().to(device))

        if self.show:
            self.visualize_batch(mask, output, debug_skeletons, debug_sampled_points)

        return output

    def visualize_batch(self, mask: torch.Tensor, output: torch.Tensor, debug_skeletons, debug_sampled_points):
        import matplotlib.pyplot as plt
        import neurite as ne
        from fengepad.analysis.plot import show_scribbles
        import numpy as np

        bs = mask.shape[0]
        tensors = []
        titles = []
        for i in range(bs):
            tensors += [
                mask[i,0].cpu(),
                debug_skeletons[i].cpu(),
                debug_sampled_points[i].cpu(),
                output[i,0].cpu()
            ]
            titles += ["Input Mask", "Skeleton", "Sampled Points", "Generated Polyline"]

        fig, axes = ne.plot.slices(
            tensors,
            titles,
            show=False,
            grid=(bs,4),
            width=4*3
        )
        axes = np.array(axes).reshape(-1)
        for i in range(bs):
            show_scribbles(output[i,0].detach().cpu(), axes[i*4 + 3])

        plt.savefig("centerline_polyline_prompt_fixed.png")
        plt.show()