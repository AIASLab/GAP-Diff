from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from compute_idx_emb import compute_idx_embedding

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--method', default=None, help='the test method')
    parser.add_argument('--data_path', default='test/ffhq/per16_aspl', help='path to input image file')
    args = parser.parse_args()
    return args

def compute_face_embedding(img_path):
    """Extract face embedding vector of given image
    Args:
        img_path (str): path to image
    Returns:
        None: no face found
        vector: return the embedding of biggest face among the all found faces
    """
    try:
        resps = DeepFace.represent(img_path = os.path.join(img_path), 
                                   model_name="ArcFace", 
                                   enforce_detection=True, 
                                   detector_backend="retinaface", 
                                   align=True)
        if resps == 1:
            # detect only 1 face
            return np.array(resps[0]["embedding"])
        else:
            # detect more than 1 faces, choose the biggest one
            resps = list(resps)
            resps.sort(key=lambda resp: resp["facial_area"]["h"]*resp["facial_area"]["w"], reverse=True)
            return np.array(resps[0]["embedding"])
    except Exception:
        # no face found
        return None

def get_precomputed_embedding(path):
    """Get face embedding by loading the path to numpy file
    Args:
        path (str): path to numpy file 
    Returns:
        vector: face embedding
    """
    return np.load(path)


def matching_score_id(image_path, avg_embedding):
    """getting the matching score between face image and precomputed embedding

    Args:
        img (2D images): images
        emb (vector): face embedding

    Returns:
        None: cannot detect face from img
        int: identity score matching
    """
    image_emb = compute_face_embedding(image_path)
    id_emb = avg_embedding
    if image_emb is None:
        return None
    image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
    ism = F.cosine_similarity(image_emb, id_emb, dim=0)
    return ism

def matching_score_genimage_id(images_path, list_id_path):
    image_list = os.listdir(images_path)
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        #print(fail_detection_count)
        #print(len(image_list))
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return None, 1


def main():
    args = parse_args()
    data_main_folder = args.data_path
    emb_main_folder = "data/test_dataset"
    subfolders = [f for f in os.listdir(data_main_folder) if os.path.isdir(os.path.join(data_main_folder, f))]
    ism_sum = 0
    fdfr_sum = 0
    count = 0
    prompts = [
        "a_photo_of_sks_person",
        "a_dslr_portrait_of_sks_person",
        "a_photo_of_sks_person_looking_at_the_mirror",
        "a_photo_of_sks_person_in_front_of_eiffel_tower",
    ]
    print("Now evaluate the effectiveness of " + args.method)
    for prompt in prompts:
        ism_sum = 0
        fdfr_sum = 0
        count = 0
        for subfolder in subfolders:
            data_subfolder_path = os.path.join(data_main_folder, subfolder)
            emb_subfolder_folder = os.path.join(emb_main_folder, subfolder)
            data_dir = os.path.join(data_subfolder_path, prompt)
            emb_dir = os.path.join(emb_subfolder_folder, "set_C")
            if os.path.isdir(data_dir) and os.path.isdir(emb_dir):
                ism, fdfr = matching_score_genimage_id(data_dir, [emb_dir])
                if ism is not None:
                    ism_sum += ism
                fdfr_sum += fdfr
                count += 1
                #print("ISM and FDR are {} and {}".format(ism, fdfr))
        print("     MEAN_FDFR and MEAN_ISM of " + prompt + " are {} and {}".format(fdfr_sum * 100 / count, ism_sum / count))
if __name__ == '__main__':
    main()

