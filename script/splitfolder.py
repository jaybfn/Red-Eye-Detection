# import libraries
import splitfolders as sf

# path
FOLDER_NAME = '../data/eye_data/'

# splitfolder
sf.ratio(FOLDER_NAME, output="../data/eye_data/training", seed=142, ratio=(0.8,0.15,0.05))