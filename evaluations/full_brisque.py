import argparse
import os
from PIL import Image
from brisque import BRISQUE

def parse_args():
    parser = argparse.ArgumentParser(description='Brisque')
    parser.add_argument('--method', default=None, help='the test method')
    parser.add_argument('--data_path', default=None, help='path to input image file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_main_folder = args.data_path
    obj = BRISQUE(url=False)
    subfolders = [f for f in os.listdir(data_main_folder) if os.path.isdir(os.path.join(data_main_folder, f))]
    
    prompts = [
        "a_photo_of_sks_person",
        "a_dslr_portrait_of_sks_person",
        "a_photo_of_sks_person_looking_at_the_mirror",
        "a_photo_of_sks_person_in_front_of_eiffel_tower",
    ]
    print("Now evaluate the effectiveness of " + args.method)
    for prompt in prompts:
        brisque_sum = 0
        count_sum = 0
        #print(">>>>>>", prompt)
        for subfolder in subfolders:
            brisque_sin = 0
            count_sin = 0
            data_subfolder_path = os.path.join(data_main_folder, subfolder)
            data_dir = os.path.join(data_subfolder_path, prompt)
            for img_name in os.listdir(data_dir):
                if "png" in img_name or "jpg" in img_name:
                    img_path = os.path.join(data_dir, img_name)
                    img = Image.open(img_path)
                    brisque_score = obj.score(img)
                    brisque_sum += brisque_score
                    brisque_sin += brisque_score
                    count_sum += 1
                    count_sin += 1               
            #print("brisque_score is {}".format(brisque_sin/count_sin))
        print("     MEAN_BRISQUE of " + prompt + " is {}".format(brisque_sum / count_sum))
        

if __name__ == '__main__':
    brisque = main()