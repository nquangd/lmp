# Prototype: generation-wise lung deposition -> scintigraphy-like planar image
# - Symmetric Weibel-style airway scaffold (simple geometry, flexible to swap in any tree)
# - Map per-generation deposition fractions to a 3D activity volume (mass-conserving)
# - Forward project with attenuation + PSF + Poisson to make an A-P planar image
#
# Notes:
# - Keep it lightweight (numpy, scipy). No external data or internet required.
# - You can replace `build_weibel_tree` output with your own subject-specific tree:
#     list of segments: dict(start, end, gen, radius_mm)
# - Tunable constants are grouped near the top.

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ------------------------------
# Tunable parameters
# ------------------------------
VOXEL_SIZE_MM = 3.0            # isotropic voxel size
GRID_SHAPE = (128, 128, 128)   # (Z, Y, X) for faster projection along Z (A-P)
MAX_GEN = 12                   # Maximum generation to simulate (conducting+acinar)
CONDUCTING_MAX_GEN = 8         # Up to this gen is treated as centerline/airway
TOTAL_ACTIVITY = 1e7           # Arbitrary total counts scale (before projection)
MU_ATTEN = 0.010               # Effective linear attenuation (1/mm) within thorax
PSF_FWHM_PIX = 3.5             # 2D Gaussian PSF FWHM in detector pixels
BACKGROUND_COUNTS = 2.0        # Uniform background counts per pixel (Poisson mean)
RNG_SEED = 7                   # For reproducibility of tree jitter/noise

np.random.seed(RNG_SEED)

# ------------------------------
# Utility
# ------------------------------
@dataclass
class Segment:
    start: np.ndarray  # 3D point in mm (z, y, x)
    end:   np.ndarray  # 3D point in mm (z, y, x)
    gen:   int         # Generation index (0 = trachea)
    r_mm:  float       # Approx airway radius in mm

def mm_to_idx(coord_mm: np.ndarray) -> np.ndarray:
    """Convert mm coordinates (z,y,x) to voxel indices (z,y,x)."""
    return np.round(coord_mm / VOXEL_SIZE_MM).astype(int)

def fwhm_to_sigma_pix(fwhm_pix: float) -> float:
    return fwhm_pix / np.sqrt(8.0 * np.log(2.0))

# ------------------------------
# Anatomical scaffold
# ------------------------------
def build_lung_mask(shape=GRID_SHAPE, voxel_mm=VOXEL_SIZE_MM) -> np.ndarray:
    """
    Two-ellipsoid lung mask in a thoracic box; lungs centered laterally.
    Returns a boolean mask (Z,Y,X).
    """
    Z, Y, X = shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    # Convert to mm coordinates centered roughly mid-lung
    center = np.array([Z/2, Y/2, X/2])
    z_mm = (zz - center[0]) * voxel_mm
    y_mm = (yy - center[1]) * voxel_mm
    x_mm = (xx - center[2]) * voxel_mm

    # Left and right ellipsoids (approximate)
    # Semi-axes in mm
    a_z, a_y, a_x = 110, 90, 60   # cranio-caudal, antero-posterior, medio-lateral radii
    # Offset to create two lobes
    x_shift = 70  # mm

    left = (z_mm/a_z)**2 + (y_mm/a_y)**2 + ((x_mm + x_shift)/a_x)**2 <= 1.0
    right = (z_mm/a_z)**2 + (y_mm/a_y)**2 + ((x_mm - x_shift)/a_x)**2 <= 1.0
    # Exclude mediastinum area (thin gap)
    mediastinum = np.abs(x_mm) < 10
    lung = (left | right) & (~mediastinum)
    return lung

def build_weibel_tree(max_gen=MAX_GEN) -> List[Segment]:
    """
    Simple symmetric Weibel-like tree in mm coordinates within thorax box.
    The geometry is heuristic but produces plausible central->peripheral spread.
    """
    # Thorax extent in mm based on grid
    Z_mm = GRID_SHAPE[0] * VOXEL_SIZE_MM
    Y_mm = GRID_SHAPE[1] * VOXEL_SIZE_MM
    X_mm = GRID_SHAPE[2] * VOXEL_SIZE_MM

    # Start at trachea near superior mediastinum
    trachea_start = np.array([Z_mm*0.25, Y_mm*0.5, X_mm*0.5])
    trachea_end   = np.array([Z_mm*0.38, Y_mm*0.5, X_mm*0.5])

    # Base radii/lengths (heuristic)
    base_radius = 6.0  # mm (trachea)
    base_len    = 25.0 # mm segment length at gen 0

    # Scaling per generation
    len_scale = 0.85
    rad_scale = 0.80
    branch_angle_deg = 28.0  # typical bifurcation angle from parent axis
    yaw_jitter_deg = 18.0    # random yaw to fill 3D space over generations

    segs: List[Segment] = [Segment(trachea_start, trachea_end, 0, base_radius)]

    # Seed main bronchi by bifurcating the trachea end
    frontier = [segs[0]]

    for g in range(1, max_gen+1):
        new_frontier = []
        for parent in frontier:
            # Parent axis
            v = parent.end - parent.start
            v = v / (np.linalg.norm(v) + 1e-8)

            # Create two child directions by rotating v with pitch & yaw
            def make_child_dir(sign: int) -> np.ndarray:
                # Pitch down cranio-caudally (increase z), fan laterally with sign
                pitch = np.deg2rad(branch_angle_deg)
                yaw = np.deg2rad(sign * (branch_angle_deg + np.random.uniform(-yaw_jitter_deg, yaw_jitter_deg)))
                # Build rotation via sequential yaw (x) and roll around y
                # For simplicity, define a canonical downwards vector and mix with offsets
                # We bias growth towards increasing z (inferior) and ±x (lateral)
                down = np.array([1.0, 0.0, 0.0])   # +z direction in (z,y,x)
                lat  = np.array([0.0, 0.0, 1.0]) * np.sign(sign)  # ±x
                ant  = np.array([0.0, 1.0, 0.0])   # +y (anterior)

                d = (np.cos(pitch)*down +
                     0.55*np.sin(pitch)*lat +
                     0.25*np.sin(pitch)*np.cos(yaw)*ant)
                d = d / (np.linalg.norm(d) + 1e-8)
                # Blend with parent direction to keep continuity
                d = (0.45*v + 0.55*d)
                return d / (np.linalg.norm(d) + 1e-8)

            child_len = base_len * (len_scale ** (g-0))
            child_rad = max(0.5, base_radius * (rad_scale ** (g)))  # floor radius

            for sgn in (-1, +1):
                d = make_child_dir(sgn)
                start = parent.end
                end   = start + d * child_len
                seg = Segment(start, end, g, child_rad)
                new_frontier.append(seg)
                segs.append(seg)
        frontier = new_frontier
        # Stop growing conducting tree after a few generations; remainder considered acinar
        if g >= CONDUCTING_MAX_GEN:
            break
    return segs

# ------------------------------
# Generation -> 3D activity mapping
# ------------------------------
def splat_line(volume, p0_mm, p1_mm, radius_mm, sigma_vox=0.6):
    """
    Rasterize a line segment from p0_mm to p1_mm by sampling points and
    adding Gaussian splats into the 3D volume. Mass-conserving via weight normalization.
    """
    p0 = p0_mm / VOXEL_SIZE_MM
    p1 = p1_mm / VOXEL_SIZE_MM
    length_vox = np.linalg.norm(p1 - p0)
    n_samples = max(2, int(length_vox * 2))  # 2 samples per voxel length

    zs = np.linspace(p0[0], p1[0], n_samples)
    ys = np.linspace(p0[1], p1[1], n_samples)
    xs = np.linspace(p0[2], p1[2], n_samples)

    Z, Y, X = volume.shape
    half = int(np.ceil(2.5 * sigma_vox))
    # Precompute a local kernel grid
    coords = np.arange(-half, half+1)
    zz_k, yy_k, xx_k = np.meshgrid(coords, coords, coords, indexing='ij')
    kernel = np.exp(-(zz_k**2 + yy_k**2 + xx_k**2)/(2*sigma_vox**2))
    kernel /= kernel.sum() + 1e-12

    for zf, yf, xf in zip(zs, ys, xs):
        zi, yi, xi = int(round(zf)), int(round(yf)), int(round(xf))
        z0, z1 = max(0, zi-half), min(Z, zi+half+1)
        y0, y1 = max(0, yi-half), min(Y, yi+half+1)
        x0, x1 = max(0, xi-half), min(X, xi+half+1)

        kz0 = z0 - (zi-half); ky0 = y0 - (yi-half); kx0 = x0 - (xi-half)
        kz1 = kernel.shape[0] - ((zi+half+1) - z1)
        ky1 = kernel.shape[1] - ((yi+half+1) - y1)
        kx1 = kernel.shape[2] - ((xi+half+1) - x1)

        volume[z0:z1, y0:y1, x0:x1] += kernel[kz0:kz1, ky0:ky1, kx0:kx1]

def build_activity_from_generations(Dg: np.ndarray,
                                    segs: List[Segment],
                                    lung_mask: np.ndarray,
                                    total_activity: float = TOTAL_ACTIVITY,
                                    conducting_max_gen: int = CONDUCTING_MAX_GEN) -> np.ndarray:
    """
    Map per-generation fractions Dg (len = MAX_GEN+1 or more) into a 3D activity volume.
    Mass is conserved: sum(activity) == total_activity * sum(Dg_in_mask)
    Conducting generations -> along centerlines; acinar generations -> diffuse parenchyma.
    """
    Z, Y, X = lung_mask.shape
    act = np.zeros((Z, Y, X), dtype=np.float64)

    # Normalize kernel per generation by where we deposit
    Dg = np.asarray(Dg, dtype=float)
    Gmax = min(len(Dg)-1, MAX_GEN)

    # Conducting: build per-gen temp volumes from splats
    for g in range(0, min(conducting_max_gen, Gmax)+1):
        gen_mass = total_activity * Dg[g]
        if gen_mass <= 0:
            continue
        tmp = np.zeros_like(act)
        # Splat each segment at this generation
        segs_g = [s for s in segs if s.gen == g]
        if not segs_g:
            continue
        # Small sigma in voxels to keep central streaks thin
        sigma_vox = 0.5 + 0.1*g
        for s in segs_g:
            splat_line(tmp, s.start, s.end, s.r_mm, sigma_vox=sigma_vox)
        ssum = tmp.sum()
        if ssum > 0:
            act += tmp * (gen_mass / ssum)

    # Acinar: diffuse into peripheral parenchyma (distance-from-center weighting)
    if Gmax > conducting_max_gen:
        # Peripheral weighting: farther from mediastinum (x=0) and nearer to pleura
        Zc, Yc, Xc = np.array(lung_mask.shape)/2.0
        zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
        x_dist = np.abs(xx - Xc)  # lateral distance
        # Approximate pleural distance via distance to mask boundary (cheap proxy)
        # Here we use inverse of a blurred mask as a crude "near pleura" weight.
        lung_float = lung_mask.astype(float)
        near_pleura = lung_float - gaussian_filter(lung_float, sigma=2.0)
        near_pleura = np.clip(near_pleura, 0, None)
        periph_w = (x_dist / x_dist.max()) + 1.5*near_pleura
        periph_w *= lung_mask
        periph_w = np.clip(periph_w, 0, None)

        wsum = periph_w.sum() + 1e-12
        for g in range(conducting_max_gen+1, Gmax+1):
            gen_mass = total_activity * Dg[g]
            if gen_mass <= 0:
                continue
            act += periph_w * (gen_mass / wsum)

    return act

# ------------------------------
# Imaging physics: A-P planar projection
# ------------------------------
def project_planar_AP(activity: np.ndarray,
                      lung_mask: np.ndarray,
                      mu_attn: float = MU_ATTEN,
                      voxel_mm: float = VOXEL_SIZE_MM,
                      psf_fwhm_pix: float = PSF_FWHM_PIX,
                      background_counts: float = BACKGROUND_COUNTS,
                      rng=np.random) -> np.ndarray:
    """
    Simple A-P planar projection:
      y(u,v) = sum_k a[z_k,y=v,x=u] * exp(-mu * depth_mm) * voxel_mm
      then 2D PSF blur + Poisson noise + background
    """
    Z, Y, X = activity.shape
    # Depth from posterior to anterior: assume posterior is high z index -> moving towards low z
    # For A-P, detector in front (anterior); path length from each voxel to detector depends on k index.
    # Use cumulative attenuation from each z to anterior surface (z=0 plane).
    dz = voxel_mm

    # Precompute attenuation factors per z
    # depth_mm for a voxel at slice z is z * dz to the detector at z=0
    z_indices = np.arange(Z)
    depth_mm = z_indices[:, None, None] * dz
    att_factors = np.exp(-mu_attn * depth_mm)  # shape (Z,1,1)

    # Line integral with attenuation weighting
    # Multiply activity by attenuation, sum along z
    weighted = activity * att_factors
    proj = weighted.sum(axis=0) * dz  # (Y,X)

    # Simple PSF blur
    sigma_pix = fwhm_to_sigma_pix(psf_fwhm_pix)
    if sigma_pix > 0:
        proj = gaussian_filter(proj, sigma=sigma_pix, mode='nearest')

    # Add background and Poisson noise
    mean_image = proj + background_counts
    noisy = rng.poisson(mean_image).astype(float)
    return noisy

# ------------------------------
# Demo
# ------------------------------
def demo():
    lung = build_lung_mask()
    segs = build_weibel_tree(max_gen=MAX_GEN)

    # Example per-generation deposition fractions (normalize to 0.25 intrathoracic fraction)
    G = MAX_GEN + 1
    g = np.arange(G)
    # Toy shape: modest central then rising peripheral
    Dg = 0.02*np.exp(-0.3*g) + 0.01*(g/(G-1))
    Dg = Dg / Dg.sum() * 0.25  # e.g., 25% of inhaled dose is intrathoracic

    activity = build_activity_from_generations(Dg, segs, lung_mask=lung, total_activity=TOTAL_ACTIVITY)

    planar_ap = project_planar_AP(activity, lung_mask=lung)

    # Quick QC
    print("Mass conservation check (should equal TOTAL_ACTIVITY * sum(Dg)):")
    print(activity.sum(), TOTAL_ACTIVITY * Dg.sum())

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # MIP for visualization (no attenuation/PSF), purely illustrative
    mip = activity.max(axis=0)
    axes[0].imshow(mip, origin='lower')
    axes[0].set_title("3D Activity MIP (Y-X)")
    axes[0].axis('off')

    axes[1].imshow(planar_ap, origin='lower')
    axes[1].set_title("Simulated Planar A-P")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

demo()