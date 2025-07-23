#%% Base Imports

import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

dataset = dm.load('asu_campus_3p5')

params = dm.ChannelParameters()
params.bs_antenna.rotation = [0, 0, -135]
dataset.set_channel_params(params)

#%% Plot converage

plt.rcdefaults()
plt.rcParams['text.color'] = 'white'           # Font color
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.facecolor'] = 'grey'     # Background outside plot
plt.rcParams['axes.facecolor'] = 'white'       # Background inside plot

attrs = ['power', 'phase', 'delay', 'aoa_az', 'aoa_el', 'aod_az', 'aod_el' ]

for attr in attrs:
    dataset[attr].plot(dpi=300)
    plt.savefig(f"dm_coverage_{attr}.png", transparent=True)

#%% Plot LoS

plt.rcdefaults()
plt.rcParams['text.color'] = 'white'           # Font color
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'       # Background inside plot

dataset.los = dataset.los.astype(float)
dataset.los[dataset.los == -1] = np.nan

dataset.los.plot(dpi=300, cbar_labels=['LoS', 'NLoS'])
plt.savefig("dm_los.png", transparent=True)

#%% Plot rays

plt.rcdefaults()
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
dataset.plot_rays(55039, dpi=300)
plt.savefig("plot_rays.png", transparent=True)

#%% Plot scene
plt.rcParams['text.color'] = 'white'
plt.rcParams['font.size'] = 16
dataset.scene.plot(dpi=300)
# (additional changing terrain color to black in scene.py)

plt.savefig("dm_scene.png", transparent=True)

#%% Doppler Sequence (Static)

plt.rcdefaults()
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['font.size'] = 16

seq_idxs1 = dm.LinearPath(dataset.rx_pos, [150, -120], [150, 80], n_steps=200).idxs
seq_idxs2 = dm.LinearPath(dataset.rx_pos, [150, 85], [-120, 85], n_steps=200).idxs
seq_idxs = np.concatenate([seq_idxs1, seq_idxs2])

n = 8
ax = dataset.scene.plot(dpi=300, proj_3D=False, title='')
ax.scatter(dataset.rx_pos[seq_idxs[::n], 0], dataset.rx_pos[seq_idxs[::n], 1],
           s=40, facecolors=(1, 0, 0, 0.5), edgecolors='black', 
           label='User Positions')
plt.legend(loc='upper right')
ax.set_facecolor((1, 1, 1, 0.5))
ax.figure.patch.set_alpha(0.0)  # fully transparent outer background
plt.savefig("dm_scene_doppler_static.png")

#%% Doppler Sequence (Dynamic - with moving terrain)

def modify_terrain(terrain, center_xy=None, size_xy=None):
    """Modify a terrain object's position and/or size.
    
    Args:
        terrain: PhysicalElement object to modify
        center_xy: Optional new (x,y) center position. If None, keeps current position.
        size_xy: Optional new (width, length) in meters. If None, keeps current size.
    """
    # Use current values if new ones not provided
    current_center = terrain.position
    current_size = np.array([terrain.bounding_box.width, terrain.bounding_box.length])
    
    center = np.array(center_xy) if center_xy is not None else current_center[:2]
    size = np.array(size_xy) if size_xy is not None else current_size
    
    # Create vertices
    z = current_center[2]  # Keep original z-coordinate
    half_width, half_length = size[0]/2, size[1]/2
    vertices = np.array([
        [center[0] - half_width, center[1] - half_length, z],  # bottom-left
        [center[0] + half_width, center[1] - half_length, z],  # bottom-right
        [center[0] + half_width, center[1] + half_length, z],  # top-right
        [center[0] - half_width, center[1] + half_length, z],  # top-left
    ])
    
    # Update the terrain
    terrain._faces = [dm.scene.Face(vertices=vertices, material_idx=terrain.faces[0].material_idx)]
    terrain.vertices = vertices
    terrain._compute_bounding_box()


plt.rcdefaults()
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
proj_3D = True
folder = f"dm_scene_doppler_dynamic2_{'3D' if proj_3D else '2D'}"
os.makedirs(folder, exist_ok=True)

s = 200 # terrain size in meters (width and length)

dataset = dm.load('asu_campus_3p5')
orig_bldgs = dataset.scene.objects[:-1]
terrain = dataset.scene.objects[-1]

min_pwr = np.nanmin(dataset.power[seq_idxs])
max_pwr = np.nanmax(dataset.power[seq_idxs])

for i, usr_idx in enumerate(seq_idxs[::4]):
    usr_pos = dataset.rx_pos[usr_idx]
    
    # Resize and move the terrain under the user
    modify_terrain(terrain, center_xy=(usr_pos[0], usr_pos[1]), size_xy=(s, s))
    
    # Select buildings within the range    
    min_x = usr_pos[0] - s/2
    max_x = usr_pos[0] + s/2
    min_y = usr_pos[1] - s/2
    max_y = usr_pos[1] + s/2
    
    s2 = 20
    is_in_range = lambda b: (min_x - s2 < b.position[0] < max_x + s2) and \
                            (min_y - s2 < b.position[1] < max_y + s2)
    bldgs_in_range = [b for b in orig_bldgs if is_in_range(b)]

    # Select interaction buildings (if outside of range)
    # u_inter = np.unique(dataset.inter_obj[usr_idx])
    # usr_inter_objs = u_inter[~np.isnan(u_inter)].astype(int)
    # bldgs_in_range_ids = [b.object_id for b in bldgs_in_range]
    # inter_bldgs = [orig_bldgs[b_id] for b_id in usr_inter_objs
    #                if b_id not in bldgs_in_range_ids + [terrain.object_id]]
    inter_bldgs = []

    # Update the scene objects
    dataset.scene.objects = bldgs_in_range + inter_bldgs + [terrain]

    # Note: CHANGES NEEDED IN visualization.py: 
    # - change plot_line zorder to 1000
    # - change plot_point rx/tx to white
    # 
    # Faster
    _, ax = dataset.plot_rays(usr_idx, proj_3D=proj_3D)
    dataset.scene.plot(dpi=300, proj_3D=proj_3D, title='', ax=ax)
    
    # Slower
    # ax = dataset.scene.plot(dpi=300, proj_3D=proj_3D, title='')
    # dataset.plot_rays(usr_idx, ax=ax, proj_3D=proj_3D)

    # Add coverage map (Matplotlib has issues with 3D plots)
    if not proj_3D:
        idxs_in_range = dm.get_idxs_with_limits(dataset.rx_pos,
                                                x_min=min_x, x_max=max_x, 
                                                y_min=min_y, y_max=max_y)
        dataset_t = dataset.subset(idxs_in_range)
        dataset_t.power.plot(ax=ax, proj_3D=proj_3D, lims=(min_pwr, max_pwr))

    ax.legend().set_visible(False)
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    if proj_3D:
        ax.set_zlim((-20, 100))
    plt.savefig(f"{folder}/{i}.png", transparent=True)
    plt.close()
    # plt.show()
    print(f"Saved {i} of {len(seq_idxs[::4])}")
    # break


#%% With Static terrain

plt.rcdefaults()
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
proj_3D = True
folder = f"dm_scene_doppler_dynamic_static_{'3D' if proj_3D else '2D'}"
os.makedirs(folder, exist_ok=True)

s = 200 # terrain size in meters (width and length)

dataset = dm.load('asu_campus_3p5')
orig_bldgs = dataset.scene.objects[:-1]
terrain = dataset.scene.objects[-1]

min_pwr = np.nanmin(dataset.power[seq_idxs])
max_pwr = np.nanmax(dataset.power[seq_idxs])

# Select buildings within the range    
min_x = -250
max_x = 250
min_y = -250
max_y = 250

s2 = 20
is_in_range = lambda b: (min_x - s2 < b.position[0] < max_x + s2) and \
                        (min_y - s2 < b.position[1] < max_y + s2)
bldgs_in_range = [b for b in orig_bldgs if is_in_range(b)]

# Resize and move the terrain under the user
modify_terrain(terrain, center_xy=((max_x+min_x)/2, (max_y+min_y)/2), 
               size_xy=(max_x-min_x, max_y-min_y))

# Update the scene objects
dataset.scene.objects = bldgs_in_range + [terrain]

for i, usr_idx in enumerate(seq_idxs[::4]):
    usr_pos = dataset.rx_pos[usr_idx]
    
    _, ax = dataset.plot_rays(usr_idx, proj_3D=proj_3D, dpi=300)
    dataset.scene.plot(proj_3D=proj_3D, title='', ax=ax)
    
    # Add coverage map (Matplotlib has issues with 3D plots)
    if not proj_3D:
        idxs_in_range = dm.get_idxs_with_limits(dataset.rx_pos,
                                                x_min=min_x, x_max=max_x, 
                                                y_min=min_y, y_max=max_y)
        dataset_t = dataset.subset(idxs_in_range)
        dataset_t.power.plot(ax=ax, proj_3D=proj_3D, lims=(min_pwr, max_pwr))

    ax.legend().set_visible(False)
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    if proj_3D:
        ax.set_zlim((-20, 100))
    plt.savefig(f"{folder}/{i}.png", transparent=True)
    plt.close()
    # plt.show()
    print(f"Saved {i} of {len(seq_idxs[::4])}")
    # break


#%% Save Dynamic Video

import subprocess

subprocess.run([
    "ffmpeg",
    "-y", # overwrite if file exists
    "-framerate", "10",
    "-i", f"{folder}/%d.png",
    "-filter_complex",
    "[0]format=rgba[fg];color=black:s=3000x2400:d=5[bg];[bg][fg]overlay=format=auto",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    f"{folder}/output.mp4"
])

# FUTURE: Read the first image size and configure the command (in 3000x2400)

# Clean up PNG files
# for file in os.listdir(folder):
#     if file.endswith('.png'):
#         os.remove(os.path.join(folder, file))

#%% Repeat for Dynamic Scene

if False:
    dyn_name = dm.convert('RT_SOURCES/asu_campus_3p5_dyn_rd', 
                          overwrite=True, vis_scene=False, print_params=False)
else:
    dyn_name = 'asu_campus_3p5_dyn_rd'

# Load Dataset for moving car and for static BS
dataset_dyn = dm.load(dyn_name, tx_sets=[1], rx_sets=[0]) # car-rxgrid
dataset_dyn_rt = dm.load(dyn_name, tx_sets=[1], rx_sets=[2]) # car-BS

#%% Plot Coverage for Dynamic Scene

plt.rcdefaults()
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'

pwr_lims = (-146, -60)  # min first, max second

save = True
folder = f"dm_scene_doppler_dynamic_car_rxgrid_2D_5R1D"
os.makedirs(folder, exist_ok=True)

n_scenes = len(dataset_dyn)
for i in range(n_scenes):
    
    plot_args = dict(scat_sz=2.2, dpi=300, figsize=(10,8), lims=pwr_lims, 
                     cbar_title='Power (dBm)')
    ax, _ = dm.plot_coverage(dataset_dyn[i].rx_pos, 
                             dataset_dyn[i].power[:,0], 
                             bs_pos=dataset_dyn[i].tx_pos[0], 
                             **plot_args)
    dataset_dyn_rt[i].plot_rays(0, ax=ax, proj_3D=False,
                                # color_strat='relative', limits=pwr_lims, 
                                color_strat='absolute', limits=pwr_lims, 
                                show_cbar=False)
    
    ax.legend().set_visible(False)
    ax.grid(False)
    
    if save:
        plt.savefig(f"{folder}/{i}.png", transparent=True)
        plt.close()
        print(f"Saved {i} of {n_scenes}")
    else:
        plt.show()
    
    # break


