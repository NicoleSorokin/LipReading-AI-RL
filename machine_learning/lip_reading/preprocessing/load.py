import tensorflow as tf
import numpy as np
import os
from crop import Crop
from align import Align

CROPPED_SIZE = (100, 50)

def load(path):
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data','videos', 's1', f'{file_name}.mpg')
    align_path = os.path.join('data', 'aligns', 's1', 'align', f'{file_name}.align')

    cropper = Crop(video_path)
    aligner = Align(align_path)

    cropper.load_frames()
    cropper.detect_landmarks()
    cropper.crop_mouth(CROPPED_SIZE, 70, 70)
    frames = cropper.get_frames()

    aligner.load_alignment()
    alignments = aligner.to_tensor()
    
    return frames, alignments

if __name__ == "__main__":
    
    #Example path
    path = './data/videos/s1/bbal6n.mpg'

    frames, alignments = load(path)
    
    print("Frames tensor shape:", frames.shape)
    print("Alignments tensor shape:", alignments.shape)