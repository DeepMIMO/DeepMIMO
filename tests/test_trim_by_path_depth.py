import deepmimo as dm

#%% Trimming by path depth test

dataset = dm.load('asu_campus_3p5')

dataset_t = dataset.trim_by_path_depth(1)

dataset.plot_coverage(dataset.los, title='Full dataset')
dataset_t.plot_coverage(dataset_t.los, title='Trimmed dataset')

#%% num interactions
dataset.plot_coverage(dataset.num_interactions[:,0], title='Number of interactions')
dataset_t.plot_coverage(dataset_t.num_interactions[:,0], title='Number of interactions')

#%% num paths
dataset.plot_coverage(dataset.num_paths, title='Number of paths')
dataset_t.plot_coverage(dataset_t.num_paths, title='Number of paths')

#%% interaction type
dataset_t.plot_coverage(dataset_t.inter[:,0], title='Interaction type')


#%% Plot rays
dataset_t.plot_rays(9)
