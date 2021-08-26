# isosceles
ISOSCLES (Iterative Self-Organizing SCene LEvel Sampling) is a hierarchical, unsupervised method for obtaining a set of image chips for training convolutional neural networks. ISOSCELES operates at two levels, first selecting representative images from a full image domain (scene_select.py) and then performing unsupervised clustering of potential samples from each of those scenes to obtain a final set of sample chips that then can be labelled or used for unsupervised training. This technique was first designed to support building extraction tasks using Maxar 4-band imagery from WorldView-2, WorldView-3, Quickbird-2, or GeoEye, but can be modified to work with other image sources.


scene_select.py		This script runs from command line and uses affinity propagation clustering to select 
image scenes that will be sampled with ISOSCELES. The csv output can be used directly with ISOSCELES if 
running on Windows, or can be converted (see helper method in training_utils) to Linux filepath format to 
run on smapper or other servers. Arguments are:

  --stats	Path to csv with image stats
  --stats_idx	Column number (start from 0) of filepaths in stats csv
  --meta         	Optional. Path to csv containing image metadata
  --meta_idx  	Column number (start from 0) of filepaths in meta csv
  --tindex       	Column name for filepaths
  --out_fn       	Full filepath for output csv
  --pref           	Optional. AP preference value to be used instead of default

Example usage:

	python scene_select.py --stats C:\Users\some_user\some_coverage_stats.csv --stats_idx 0 
	--meta C:\Users\some_user\some_coverage_meta.csv --meta_idx 0 --tindex filepath 
	-- out_fn C:\Users\some_user\some_coverage_exemplars.csv


isosceles.py     Input is a text file with each line being the full path to one image. In the destination folder, 
a subfolder for each exemplar scene will be created and exemplar chips copied. Additionally, 
a csv will be output for each exemplar scene along with the image features calculated by the script.

  --wd		Working directory to store temp raster
  --list		Path to text file listing images to process
  --dst		Top level folder for output 
  --fn		Prefix for output statistics files [e.g. ‘ethiopia’]
  --n_proc	Number of worker processes to use
  --pref		Optional. AP preference value to be used instead of default. 

Example usage:

	python isosceles_mp.py --wd /home/some_user --list /home/some_user/some_exemplar_list.txt 
	--dst /mnt/GATES/UserDirs/some_user/some_project_exemplars --fn some_project --n_proc 12
