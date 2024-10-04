import cv2
import argparse
import os
from FaceImageQuality.face_image_quality import SER_FIQ

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--method', default=None, help='the test method')
    parser.add_argument('--data_path', help='path to input image file')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_main_folder = args.data_path
    ser_fiq = SER_FIQ(gpu=0)
    subfolders = [f for f in os.listdir(data_main_folder) if os.path.isdir(os.path.join(data_main_folder, f))]

    prompts = [
        "a_photo_of_sks_person",
        "a_dslr_portrait_of_sks_person",
        "a_photo_of_sks_person_looking_at_the_mirror",
        "a_photo_of_sks_person_in_front_of_eiffel_tower",
    ]
    print("Now evaluate the effectiveness of " + args.method)
    for prompt in prompts:
        ser_sum = 0
        count_sum = 0
        for subfolder in subfolders:
            ser_sin = 0
            count_sin = 0
            data_subfolder_path = os.path.join(data_main_folder, subfolder)
            data_dir = os.path.join(data_subfolder_path, prompt)
            for img_name in os.listdir(data_dir):
                if "png" in img_name or "jpg" in img_name:
                    img_path = os.path.join(data_dir, img_name)
                    img = cv2.imread(img_path)
                    aligned_img = ser_fiq.apply_mtcnn(img)
                    if aligned_img is not None:
                        score = ser_fiq.get_score(aligned_img, T=100)
                        ser_sin += score
                        ser_sum += score
                    count_sum += 1
                    count_sin += 1               
            #print("ser_score is {}".format(ser_sin/count_sin))
        print("     MEAN_SEE_FIQ of " + prompt + " is {}".format(ser_sum / count_sum))
        

if __name__ == '__main__':
    brisque = main()